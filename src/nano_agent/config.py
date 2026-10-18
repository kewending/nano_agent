import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file automatically when config is imported
load_dotenv()

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

@dataclass
class ModelConfig:
    """Encapsulates connection details for an LLM provider."""

    api_key: str
    base_url: str | None
    model: str


def get_model_config(provider: str = "llama") -> ModelConfig:
    """
    Retrieve model configuration for the requested provider ('llama' or 'gemini'/'openai').
    Validates credentials only for the provider being used.
    """
    provider = provider.lower()

    if provider == "llama":
        api_key = os.getenv("LLAMA_API_KEY")
        base_url = os.getenv("LLAMA_BASE_URL")
        model = os.getenv("LLAMA_MODEL")
        if not base_url:
            raise ValueError("LLAMA_BASE_URL is missing in .env")
        if not model:
            raise ValueError("LLAMA_MODEL is missing in .env")
        return ModelConfig(api_key=api_key, base_url=base_url, model=model)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL")
        model = os.getenv("GEMINI_MODEL")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env")
        if not base_url:
            raise ValueError("GEMINI_BASE_URL is missing in .env")
        if not model:
            raise ValueError("GEMINI_MODEL is missing in .env")
        return ModelConfig(api_key=api_key, base_url=base_url, model=model)

    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose 'llama' or 'gemini'."
        )


def create_client(config: ModelConfig) -> OpenAI:
    """Factory function to build an OpenAI client from a ModelConfig."""
    return OpenAI(api_key=config.api_key, base_url=config.base_url)