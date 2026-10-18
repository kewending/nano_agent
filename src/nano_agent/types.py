from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMetadata:
    """Tracks token usage across requests and conversation turns."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    thinking_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "UsageMetadata":
        """Factory method to construct UsageMetadata directly from an OpenAI ChatCompletion response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return cls()

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        # Extract prompt cache details if supported by the provider
        cached_tokens = 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        # Extract reasoning/thinking tokens (e.g. OpenAI o-series, or Gemini hidden thought overhead)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if completion_details:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        if reasoning_tokens == 0 and total_tokens > (prompt_tokens + completion_tokens):
            # Fallback calculation for providers like Gemini where thinking tokens are in total_tokens
            reasoning_tokens = total_tokens - (prompt_tokens + completion_tokens)

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=reasoning_tokens,
        )

    def add(self, other: "UsageMetadata") -> None:
        """Accumulate token counts across multiple turns or steps."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.thinking_tokens += other.thinking_tokens

    def __str__(self) -> str:
        """Formatted display string for user or terminal output."""
        cache_str = f" ({self.cached_tokens} cached)" if self.cached_tokens > 0 else ""
        think_str = f" ({self.thinking_tokens} thinking)" if self.thinking_tokens > 0 else ""
        return (
            f"📊 Tokens: {self.prompt_tokens} in{cache_str} | "
            f"{self.completion_tokens} out{think_str} | "
            f"{self.total_tokens} total"
        )