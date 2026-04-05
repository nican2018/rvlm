"""
Caching service for RVLM local inference.

Provides three layers of caching to eliminate redundant computation during
the iterative generate-execute loop:

1. **VisionFeatureCache** — caches the vision encoder output (pixel-to-embedding)
   keyed by a SHA-256 hash of the raw image bytes.  Avoids re-running the
   expensive ViT forward pass when the same image appears across turns.

2. **KVCacheManager** — retains the KV-cache for the shared prompt prefix
   (system prompt + image tokens) so that only the new per-iteration suffix
   is computed on each turn.  On Gemma 4 with 4 BraTS modalities this can
   save >1 000 redundant token-positions per iteration.

3. **ChunkedPrefill** — splits long prefill sequences into GPU-friendly chunks
   to avoid OOM on memory-constrained devices.  Each chunk's KV-cache is
   accumulated incrementally and the final logits come from the last chunk.

All three components are opt-in and backward-compatible: setting
``enable_cache=True`` on ``HFLocalClient`` activates them transparently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheStats:
    """Lightweight telemetry for the caching service."""

    vision_hits: int = 0
    vision_misses: int = 0
    kv_reuses: int = 0
    kv_rebuilds: int = 0
    prefill_chunks: int = 0
    total_saved_tokens: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "vision_cache": {"hits": self.vision_hits, "misses": self.vision_misses},
            "kv_cache": {"reuses": self.kv_reuses, "rebuilds": self.kv_rebuilds},
            "prefill_chunks": self.prefill_chunks,
            "total_saved_tokens": self.total_saved_tokens,
        }


class VisionFeatureCache:
    """Cache vision-encoder outputs keyed by image content hash.

    Stores the dense embedding tensor returned by the vision encoder so that
    repeated ``describe_image()`` / ``llm_query_with_images()`` calls on the
    same image skip the ViT forward pass entirely.

    Thread-safe for read; writes are append-only (no eviction needed for the
    small number of images in a typical RVLM session — ≤ 20).
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}  # hash → tensor
        self.stats = CacheStats()

    @staticmethod
    def _hash_image(pil_image: Any) -> str:
        """SHA-256 of the raw RGB bytes — deterministic for identical pixels."""
        raw = pil_image.convert("RGB").tobytes()
        return hashlib.sha256(raw).hexdigest()

    def get(self, pil_image: Any) -> Any | None:
        """Return cached vision features or None."""
        key = self._hash_image(pil_image)
        features = self._store.get(key)
        if features is not None:
            self.stats.vision_hits += 1
        else:
            self.stats.vision_misses += 1
        return features

    def put(self, pil_image: Any, features: Any) -> None:
        """Store vision features for an image."""
        key = self._hash_image(pil_image)
        self._store[key] = features

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


class KVCacheManager:
    """Manages a rolling KV-cache across RVLM iterations.

    The RVLM loop sends the same *prefix* (system prompt + image tokens) on
    every turn, appending only the new iteration suffix.  This manager:

    1. After the first forward pass, snapshots the KV-cache at the boundary
       between the shared prefix and the per-turn suffix.
    2. On subsequent turns, restores the snapshot so that only the new suffix
       tokens need a forward pass.

    Compatible with ``transformers.DynamicCache`` (Gemma 4, Llama 3, etc.).
    """

    def __init__(self) -> None:
        self._prefix_cache: Any | None = None  # DynamicCache snapshot
        self._prefix_len: int = 0
        self.stats = CacheStats()

    @property
    def has_prefix(self) -> bool:
        return self._prefix_cache is not None

    def save_prefix(self, past_key_values: Any, prefix_len: int, torch_module: Any) -> None:
        """Snapshot the KV-cache up to ``prefix_len`` token positions.

        Args:
            past_key_values: The ``past_key_values`` returned by the model.
            prefix_len: Number of tokens that form the shared prefix.
            torch_module: The ``torch`` module (avoids top-level import).
        """
        self._prefix_len = prefix_len
        # Deep-clone each (key, value) pair so the snapshot is independent of
        # in-place mutations during later generation steps.
        self._prefix_cache = self._clone_kv(past_key_values, prefix_len, torch_module)
        self.stats.kv_rebuilds += 1

    def restore_prefix(self, torch_module: Any) -> tuple[Any, int]:
        """Return a fresh clone of the prefix KV-cache.

        Returns:
            (past_key_values, prefix_len) — ready to pass into ``model.generate()``.
        """
        if self._prefix_cache is None:
            raise RuntimeError("No prefix cache saved yet — call save_prefix first.")
        self.stats.kv_reuses += 1
        cloned = self._clone_kv(self._prefix_cache, self._prefix_len, torch_module)
        return cloned, self._prefix_len

    def invalidate(self) -> None:
        """Discard the cached prefix (e.g. when images change)."""
        self._prefix_cache = None
        self._prefix_len = 0

    @staticmethod
    def _clone_kv(past_key_values: Any, max_len: int, torch_module: Any) -> Any:
        """Clone a DynamicCache / tuple-of-tuples up to ``max_len`` positions."""
        # transformers.DynamicCache stores (key, value) per layer
        if hasattr(past_key_values, "key_cache"):
            # DynamicCache — new transformers ≥ 4.38
            from transformers import DynamicCache

            clone = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx][:, :, :max_len, :].clone()
                v = past_key_values.value_cache[layer_idx][:, :, :max_len, :].clone()
                clone.update(k, v, layer_idx)
            return clone

        # Legacy tuple-of-tuples format
        cloned_layers = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            cloned_layers.append((k[:, :, :max_len, :].clone(), v[:, :, :max_len, :].clone()))
        return tuple(cloned_layers)


def chunked_prefill(
    model: Any,
    input_ids: Any,
    attention_mask: Any | None = None,
    pixel_values: Any | None = None,
    chunk_size: int = 512,
    torch_module: Any = None,
    **extra_inputs: Any,
) -> tuple[Any, Any]:
    """Run prefill in fixed-size chunks to bound peak GPU memory.

    Processes the input_ids sequence in windows of ``chunk_size`` tokens,
    accumulating the KV-cache incrementally.  Vision inputs (``pixel_values``)
    are included only in the first chunk (where the image tokens reside).

    Args:
        model: A HuggingFace model with a ``forward()`` that accepts
               ``past_key_values`` and ``use_cache=True``.
        input_ids: ``(1, seq_len)`` token IDs.
        attention_mask: Optional ``(1, seq_len)`` mask.
        pixel_values: Optional vision input tensor (included in first chunk only).
        chunk_size: Maximum tokens per forward pass.
        torch_module: The ``torch`` module.
        **extra_inputs: Additional model inputs (e.g. ``image_sizes``).

    Returns:
        (logits_last_chunk, past_key_values) — the logits from the final
        chunk and the accumulated KV-cache covering the entire sequence.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: N811

    seq_len = input_ids.shape[1]

    if seq_len <= chunk_size:
        # Short enough — single pass, no chunking needed
        fwd_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": True,
        }
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            fwd_kwargs["pixel_values"] = pixel_values
        for k, v in extra_inputs.items():
            fwd_kwargs[k] = v

        with torch_module.inference_mode():
            outputs = model(**fwd_kwargs)
        return outputs.logits, outputs.past_key_values

    past_key_values = None
    logits = None
    chunks_processed = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]

        fwd_kwargs = {
            "input_ids": chunk_ids,
            "use_cache": True,
        }

        if past_key_values is not None:
            fwd_kwargs["past_key_values"] = past_key_values

        # Full attention mask up to current position
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask[:, :end]

        # Vision inputs only in the first chunk
        if start == 0 and pixel_values is not None:
            fwd_kwargs["pixel_values"] = pixel_values
            for k, v in extra_inputs.items():
                fwd_kwargs[k] = v

        with torch_module.inference_mode():
            outputs = model(**fwd_kwargs)

        past_key_values = outputs.past_key_values
        logits = outputs.logits
        chunks_processed += 1

    return logits, past_key_values


@dataclass
class InferenceCache:
    """Unified caching service aggregating vision, KV, and prefill caches.

    This is the main entry-point used by ``HFLocalClient`` when
    ``enable_cache=True``.

    Usage::

        cache = InferenceCache(chunk_size=512)

        # Vision feature caching
        features = cache.vision.get(pil_img)
        if features is None:
            features = run_vit(pil_img)
            cache.vision.put(pil_img, features)

        # KV-cache prefix reuse
        if cache.kv.has_prefix:
            past, prefix_len = cache.kv.restore_prefix(torch)
        else:
            logits, past = chunked_prefill(model, ids, chunk_size=cache.chunk_size)
            cache.kv.save_prefix(past, prefix_len, torch)

        print(cache.summary())
    """

    chunk_size: int = 512
    vision: VisionFeatureCache = field(default_factory=VisionFeatureCache)
    kv: KVCacheManager = field(default_factory=KVCacheManager)
    _enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def clear(self) -> None:
        """Drop all cached state."""
        self.vision.clear()
        self.kv.invalidate()

    def summary(self) -> dict[str, Any]:
        """Merge stats from all sub-caches."""
        stats: dict[str, Any] = {}
        stats.update(self.vision.stats.summary())
        stats.update(self.kv.stats.summary())
        stats["chunk_size"] = self.chunk_size
        stats["vision_entries"] = len(self.vision)
        return stats
