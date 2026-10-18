from typing import Any

class ConversationMemory:
    """Manages multi-turn conversation history for LLM interactions."""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_user_message(self, content: str) -> None:
        """Append a user message to the context."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        """
        Append an assistant response.
        Note: An assistant message can have content=None when it triggers tool calls.
        """
        message: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            message["content"] = content
        if tool_calls is not None:
            # Ensure tool calls are clean dictionaries for all providers (OpenAI, Gemini, llama-server)
            message["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in tool_calls
            ]
        self.messages.append(message)

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        """Append the execution result of a tool call."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy of messages to prevent accidental external mutations."""
        return [dict(m) for m in self.messages]

    def clear(self) -> None:
        """Reset conversation history while preserving the system prompt."""
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def __len__(self) -> int:
        """Return total number of messages stored."""
        return len(self.messages)