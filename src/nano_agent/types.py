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

@dataclass
class ContextBreakdown:
    system_prompt_tokens: int
    tools_schema_tokens: int
    user_tokens: int
    assistant_tokens: int
    tool_calls_tokens: int
    tool_results_tokens: int
    total_tokens: int
    context_limit: int
    @property
    def conversation_tokens(self) -> int:
        return self.user_tokens + self.assistant_tokens
    @property
    def tool_use_tokens(self) -> int:
        return self.tool_calls_tokens + self.tool_results_tokens
    @property
    def usage_percentage(self) -> float:
        return (self.total_tokens / self.context_limit * 100) if self.context_limit > 0 else 0.0
    
    def __str__(self) -> str:
        """Formatted display string for user or terminal output."""
        def pct(val: int) -> str:
            return f"{(val / self.total_tokens * 100):5.1f}%" if self.total_tokens > 0 else "  0.0%"
        # Progress bar (20 chars)
        bar_width = 24
        filled = int(min(1.0, self.total_tokens / max(1, self.context_limit)) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines = [
            "┌─ Context Usage Breakdown ────────────────────────────────────┐",
            f"│ Context Window: {self.total_tokens:,} / {self.context_limit:,} tokens ({self.usage_percentage:.1f}%)",
            f"│ [{bar}]",
            "├─────────────────────────────┬──────────────┬─────────────────┤",
            "│ Component                   │ Tokens       │ Share           │",
            "├─────────────────────────────┼──────────────┼─────────────────┤",
            f"│ ⚙️  System Prompt            │ {self.system_prompt_tokens:>10,}   │ {pct(self.system_prompt_tokens):>13}   │",
            f"│ 🛠️  Tool Registry (Schemas)  │ {self.tools_schema_tokens:>10,}   │ {pct(self.tools_schema_tokens):>13}   │",
            f"│ 💬 Conversation             │ {self.conversation_tokens:>10,}   │ {pct(self.conversation_tokens):>13}   │",
            f"│    ├─ User messages         │ {self.user_tokens:>10,}   │ {pct(self.user_tokens):>13}   │",
            f"│    └─ Assistant text        │ {self.assistant_tokens:>10,}   │ {pct(self.assistant_tokens):>13}   │",
            f"│ 🔧 Tool Invocations         │ {self.tool_use_tokens:>10,}   │ {pct(self.tool_use_tokens):>13}   │",
            f"│    ├─ Tool calls            │ {self.tool_calls_tokens:>10,}   │ {pct(self.tool_calls_tokens):>13}   │",
            f"│    └─ Tool outputs (results)│ {self.tool_results_tokens:>10,}   │ {pct(self.tool_results_tokens):>13}   │",
            "└─────────────────────────────┴──────────────┴─────────────────┘",
        ]
        return "\n".join(lines)