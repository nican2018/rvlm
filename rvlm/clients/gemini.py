import base64
import mimetypes
import os
from collections import defaultdict
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rvlm.clients.base_lm import BaseLM
from rvlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiClient(BaseLM):
    """
    LM Client for running models with the Google Gemini API.
    Uses the official google-genai SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = "gemini-2.5-flash",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if api_key is None:
            api_key = DEFAULT_GEMINI_API_KEY

        if api_key is None:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        # Last call tracking
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        contents, system_instruction = self._prepare_contents(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Gemini client.")

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        self._track_cost(response, model)
        return response.text

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        contents, system_instruction = self._prepare_contents(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Gemini client.")

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        # google-genai SDK supports async via aio interface
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        self._track_cost(response, model)
        return response.text

    def _prepare_contents(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[types.Content] | str, str | None]:
        """Prepare contents and extract system instruction for Gemini API."""
        system_instruction = None

        if isinstance(prompt, str):
            return prompt, None

        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Convert OpenAI-style messages to Gemini format
            contents = []
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    # Gemini handles system instruction separately
                    system_instruction = content if isinstance(content, str) else str(content)
                elif role in ("user", "assistant"):
                    gemini_role = "model" if role == "assistant" else "user"
                    parts = self._content_to_parts(content)
                    contents.append(types.Content(role=gemini_role, parts=parts))
                else:
                    parts = self._content_to_parts(content)
                    contents.append(types.Content(role="user", parts=parts))

            return contents, system_instruction

        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _content_to_parts(self, content: str | list[dict[str, Any]]) -> list[types.Part]:
        """Convert OpenAI-style content (string or multimodal list) to Gemini Parts."""
        if isinstance(content, str):
            return [types.Part(text=content)]

        parts: list[types.Part] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(types.Part(text=item["text"]))
            elif item.get("type") == "image_url":
                image_url = item["image_url"]["url"]
                if image_url.startswith("data:"):
                    # data:image/jpeg;base64,<data> → inline_data
                    header, b64_data = image_url.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=base64.b64decode(b64_data),
                            )
                        )
                    )
                else:
                    # HTTP(S) URL → download and send as inline_data
                    resp = requests.get(
                        image_url,
                        timeout=30,
                        headers={"User-Agent": "rvlm/0.1 (image-fetch)"},
                    )
                    resp.raise_for_status()
                    mime_type = resp.headers.get("Content-Type", "").split(";")[0]
                    if not mime_type or not mime_type.startswith("image/"):
                        mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=resp.content,
                            )
                        )
                    )
        return parts

    def _track_cost(self, response: types.GenerateContentResponse, model: str):
        self.model_call_counts[model] += 1

        # Extract token usage from response
        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0

            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens
            self.model_total_tokens[model] += input_tokens + output_tokens

            # Track last call for handler to read
            self.last_prompt_tokens = input_tokens
            self.last_completion_tokens = output_tokens
        else:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0

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
