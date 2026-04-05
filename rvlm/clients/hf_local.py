"""
HuggingFace Local client — runs models locally via transformers.

Supports vision-language models (e.g. Gemma 4, MedGemma, LLaVA) with
AutoModelForImageTextToText, and text-only models with AutoModelForCausalLM.

Usage:
    client = HFLocalClient(model_name="google/gemma-4-26B-A4B-it")
    response = client.completion("Describe this image in detail.")
"""

import base64
import os
import ssl
from collections import defaultdict
from io import BytesIO
from typing import Any

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
            model_name, token=hf_token, trust_remote_code=trust_remote_code,
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

        prompt_len = inputs["input_ids"].shape[-1]

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

        # Track usage
        self._track_usage(model_name, prompt_len, output_len)

        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

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
        from PIL import Image

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
    def _load_image(source: str) -> "Image.Image":
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
