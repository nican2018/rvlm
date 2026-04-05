"""
HuggingFace Local client — runs models locally via transformers.

Supports vision-language models (e.g. Gemma 4, MedGemma, LLaVA) with
AutoModelForImageTextToText, and text-only models with AutoModelForCausalLM.

Includes an optional three-layer caching service (``enable_cache=True``):

- **Vision feature cache** — avoids re-running the ViT encoder on the same
  image across RVLM iterations.
- **KV-cache prefix reuse** — retains the KV-cache for the shared prompt
  prefix (system prompt + images) so that only the new per-turn suffix
  needs a forward pass.
- **Chunked prefill** — splits long sequences into GPU-friendly chunks to
  avoid OOM on memory-constrained devices.

Usage:
    client = HFLocalClient(model_name="google/gemma-4-26B-A4B-it", enable_cache=True)
    response = client.completion("Describe this image in detail.")
"""

from __future__ import annotations

import base64
import os
import ssl
from collections import defaultdict
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

from rvlm.clients.base_lm import BaseLM
from rvlm.core.types import ModelUsageSummary, UsageSummary

# ------------------------------------------------------------------ #
#  Singleton cache — HF model loading is expensive, so we cache the   #
#  (processor, model, torch) tuple keyed by (model_name, vision).     #
#  This prevents reloading when get_client() is called repeatedly     #
#  (e.g. inside RLM._spawn_completion_context).                       #
# ------------------------------------------------------------------ #
_MODEL_CACHE: dict[tuple[str, bool], tuple[Any, Any, Any]] = {}


def _apply_ssl_workaround() -> None:
    """Disable SSL verification for HuggingFace Hub downloads.

    Needed on machines behind corporate proxies or with self-signed certs.
    Controlled by the env var ``HF_HUB_DISABLE_SSL`` or ``CURL_CA_BUNDLE=""``.
    """
    if os.getenv("HF_HUB_DISABLE_SSL", "").lower() in ("1", "true", "yes"):
        # Tell requests / urllib3 to skip verification
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
        os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
        # Monkey-patch the default SSL context so httpx also skips
        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


class HFLocalClient(BaseLM):
    """
    LM Client for running models locally via HuggingFace transformers.

    Supports both text-only and vision-language models.
    For vision models, images embedded as OpenAI-style data URIs in the prompt
    are automatically decoded and passed to the model.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-26B-A4B-it",
        max_new_tokens: int = 2048,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        hf_token: str | None = None,
        vision: bool = True,
        trust_remote_code: bool = False,
        enable_cache: bool = False,
        prefill_chunk_size: int = 512,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            max_new_tokens: Max tokens to generate per completion.
            torch_dtype: Torch dtype for model weights ("auto", "float16", "bfloat16").
            device_map: Device placement strategy ("auto", "cpu", "cuda:0", etc.).
            hf_token: HuggingFace token for gated/private models.
            vision: If True, load as a vision-language model (AutoModelForImageTextToText).
                    If False, load as text-only (AutoModelForCausalLM).
            trust_remote_code: Whether to trust remote code in the model repo.
            enable_cache: If True, activate the three-layer inference cache:
                          vision feature cache, KV-cache prefix reuse, and
                          chunked prefill.  Recommended for RVLM multi-iteration
                          loops where the same images are analysed repeatedly.
            prefill_chunk_size: Maximum tokens per forward-pass chunk during
                                prefill (only used when enable_cache=True).
        """
        super().__init__(model_name=model_name, **kwargs)

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "hf_local backend requires 'transformers' and 'torch' to be installed.\n"
                "For vision models (e.g. Gemma 4), install torchvision from the same source "
                "as PyTorch (matching CUDA/CPU), then transformers and accelerate.\n"
                "  https://pytorch.org/get-started/locally/"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.vision = vision

        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        # Apply SSL workaround before any HF Hub calls
        _apply_ssl_workaround()

        # Use cached model if available (avoids expensive re-loads)
        cache_key = (model_name, vision)
        if cache_key in _MODEL_CACHE:
            self.processor, self.model, _ = _MODEL_CACHE[cache_key]
        else:
            self.processor, self.model = self._load_model(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hf_token=hf_token,
                vision=vision,
                trust_remote_code=trust_remote_code,
            )
            _MODEL_CACHE[cache_key] = (self.processor, self.model, torch)

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        # Inference cache (vision features, KV-prefix, chunked prefill)
        self._cache_enabled = enable_cache
        self._cache: Any = None
        if enable_cache:
            from rvlm.cache import InferenceCache

            self._cache = InferenceCache(chunk_size=prefill_chunk_size)

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_model(
        model_name: str,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        hf_token: str | None = None,
        vision: bool = True,
        trust_remote_code: bool = False,
    ) -> tuple[Any, Any]:
        """Load processor + model from HuggingFace (or local path).

        Returns (processor, model).
        """
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        if vision:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )

        # Suppress default max_length warnings
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None

        return processor, model

    # ------------------------------------------------------------------ #
    #  BaseLM interface                                                    #
    # ------------------------------------------------------------------ #

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Run a synchronous completion.

        Accepts:
        - A plain string prompt (text-only).
        - An OpenAI-style message list (may contain multimodal content with
          image_url entries that use data-URI or http(s) URLs).
        """
        messages, pil_images = self._prepare_messages(prompt)
        return self._generate(messages, pil_images, model)

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """Async wrapper — runs synchronously (local model, no I/O wait)."""
        return self.completion(prompt, model)

    # ------------------------------------------------------------------ #
    #  Generation core                                                     #
    # ------------------------------------------------------------------ #

    def _generate(
        self,
        messages: list[dict[str, Any]],
        pil_images: list[Any],
        model: str | None = None,
    ) -> str:
        model_name = model or self.model_name

        # Apply chat template
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --- Vision feature cache: reuse ViT embeddings for seen images ---
        cached_pixel_values = None
        if self._cache_enabled and self._cache is not None and pil_images:
            cached_pixel_values = self._resolve_vision_features(pil_images)

        # Tokenize — include images only if present
        if pil_images:
            inputs = self.processor(
                text=chat_text,
                images=pil_images,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=chat_text,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Substitute cached vision features if available
        if cached_pixel_values is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cached_pixel_values.to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]

        # --- Cached generation path (KV-prefix reuse + chunked prefill) ---
        if self._cache_enabled and self._cache is not None:
            output_text = self._generate_with_cache(inputs, prompt_len, model_name)
            return output_text

        # --- Standard path (no cache) ---
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
            )

        new_tokens = generated[0][prompt_len:]
        output_len = len(new_tokens)
        self._track_usage(model_name, prompt_len, output_len)

        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _generate_with_cache(
        self,
        inputs: dict[str, Any],
        prompt_len: int,
        model_name: str,
    ) -> str:
        """Generation path that uses the InferenceCache for KV-prefix reuse
        and chunked prefill.

        Flow:
        1. If the KV-cache manager has a saved prefix and the current prompt
           starts with the same prefix, restore the KV snapshot and only
           forward the new suffix tokens.
        2. Otherwise, run a (potentially chunked) prefill to build the cache
           from scratch, then save the prefix snapshot for future turns.
        3. Feed the resulting KV-cache + suffix into ``model.generate()``.
        """
        from rvlm.cache import chunked_prefill

        cache = self._cache
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        seq_len = input_ids.shape[1]

        # Separate extra vision kwargs (image_sizes, etc.)
        extra_keys = {k for k in inputs if k not in ("input_ids", "attention_mask", "pixel_values")}
        extra_inputs = {k: inputs[k] for k in extra_keys}

        past_key_values = None

        if cache.kv.has_prefix:
            # Restore the KV-cache for the shared prefix
            past_key_values, prefix_len = cache.kv.restore_prefix(self.torch)
            saved_tokens = min(prefix_len, seq_len)
            cache.kv.stats.total_saved_tokens += saved_tokens

            if saved_tokens < seq_len:
                # Forward only the new suffix
                suffix_ids = input_ids[:, saved_tokens:]
                suffix_mask = attention_mask[:, :seq_len] if attention_mask is not None else None

                with self.torch.inference_mode():
                    outputs = self.model(
                        input_ids=suffix_ids,
                        attention_mask=suffix_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
        else:
            # First call — run chunked prefill and save prefix
            _, past_key_values = chunked_prefill(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                chunk_size=cache.chunk_size,
                torch_module=self.torch,
                **extra_inputs,
            )
            cache.kv.save_prefix(past_key_values, seq_len, self.torch)

        # Generate from the cached KV state
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
            )

        new_tokens = generated[0][prompt_len:]
        output_len = len(new_tokens)
        self._track_usage(model_name, prompt_len, output_len)

        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _resolve_vision_features(self, pil_images: list[Any]) -> Any | None:
        """Check the vision feature cache for each image; run the ViT encoder
        only for unseen images.

        Returns a stacked pixel-feature tensor suitable for substitution into
        the model inputs, or None if caching cannot be applied.
        """
        if self._cache is None:
            return None

        cache = self._cache
        all_cached = True
        features_list: list[Any] = []

        for img in pil_images:
            feat = cache.vision.get(img)
            if feat is not None:
                features_list.append(feat)
            else:
                all_cached = False
                break

        if all_cached and features_list:
            return self.torch.cat(features_list, dim=0)

        # Run the vision encoder for all images and cache the results.
        # We process all images together (matching the processor's expected
        # batching) and then split per-image for caching.
        dummy_inputs = self.processor(
            text="<placeholder>",
            images=pil_images,
            return_tensors="pt",
        )
        if "pixel_values" in dummy_inputs:
            pv = dummy_inputs["pixel_values"].to(self.model.device)
            # Cache per-image slices (assumes first dim is batch)
            if pv.shape[0] == len(pil_images):
                for i, img in enumerate(pil_images):
                    cache.vision.put(img, pv[i : i + 1])
            return pv

        return None

    # ------------------------------------------------------------------ #
    #  Message / image preparation                                         #
    # ------------------------------------------------------------------ #

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Convert an OpenAI-style prompt into HF chat messages + PIL images.

        Returns:
            (messages, pil_images) — messages have {"type": "image"} placeholders
            in the order matching pil_images.
        """

        pil_images: list[Image.Image] = []

        # Plain string → simple user message
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return messages, pil_images

        # OpenAI-style message list
        hf_messages: list[dict[str, Any]] = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue

            # Multimodal content list
            hf_parts: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "text":
                    hf_parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    pil_img = self._load_image(url)
                    pil_images.append(pil_img)
                    hf_parts.append({"type": "image"})
                elif part.get("type") == "image":
                    # Already in HF format (used internally)
                    hf_parts.append(part)

            hf_messages.append({"role": role, "content": hf_parts})

        return hf_messages, pil_images

    @staticmethod
    def _load_image(source: str) -> Image.Image:
        """Load a PIL Image from a data-URI, HTTP URL, or local file path."""
        from PIL import Image

        if source.startswith("data:"):
            _, b64_data = source.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")

        if source.startswith(("http://", "https://")):
            import requests

            resp = requests.get(source, timeout=60, headers={"User-Agent": "rvlm/0.1"})
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")

        # Local file
        return Image.open(source).convert("RGB")

    # ------------------------------------------------------------------ #
    #  Usage tracking                                                      #
    # ------------------------------------------------------------------ #

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += input_tokens
        self.model_output_tokens[model] += output_tokens
        self.last_prompt_tokens = input_tokens
        self.last_completion_tokens = output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )

    # ------------------------------------------------------------------ #
    #  Cache management                                                    #
    # ------------------------------------------------------------------ #

    def get_cache_stats(self) -> dict[str, Any]:
        """Return a summary of cache hit/miss statistics.

        Returns an empty dict when caching is disabled.
        """
        if self._cache is not None:
            return self._cache.summary()
        return {}

    def clear_cache(self) -> None:
        """Drop all cached vision features and KV-cache state."""
        if self._cache is not None:
            self._cache.clear()
