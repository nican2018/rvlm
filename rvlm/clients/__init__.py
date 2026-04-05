from typing import Any

from dotenv import load_dotenv

from rvlm.clients.base_lm import BaseLM
from rvlm.core.types import ClientBackend

load_dotenv()


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """
    Route a backend key to the appropriate LM client.

    Supported backends: openai, vllm, portkey, openrouter, vercel, litellm,
    anthropic, azure_openai, gemini, hf_local.
    """
    if backend == "openai":
        from rvlm.clients.openai import OpenAIClient

        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        from rvlm.clients.openai import OpenAIClient

        assert "base_url" in backend_kwargs, (
            "base_url is required to be set to local vLLM server address for vLLM"
        )
        return OpenAIClient(**backend_kwargs)
    elif backend == "portkey":
        from rvlm.clients.portkey import PortkeyClient

        return PortkeyClient(**backend_kwargs)
    elif backend == "openrouter":
        from rvlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "vercel":
        from rvlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "litellm":
        from rvlm.clients.litellm import LiteLLMClient

        return LiteLLMClient(**backend_kwargs)
    elif backend == "anthropic":
        from rvlm.clients.anthropic import AnthropicClient

        return AnthropicClient(**backend_kwargs)
    elif backend == "gemini":
        from rvlm.clients.gemini import GeminiClient

        return GeminiClient(**backend_kwargs)
    elif backend == "hf_local":
        from rvlm.clients.hf_local import HFLocalClient

        return HFLocalClient(**backend_kwargs)
    elif backend == "azure_openai":
        from rvlm.clients.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(**backend_kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: ['openai', 'vllm', 'portkey', 'openrouter', 'litellm', 'anthropic', 'azure_openai', 'gemini', 'hf_local', 'vercel']"
        )



