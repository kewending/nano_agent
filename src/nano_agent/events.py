import time
from dataclasses import asdict, dataclass, field, KW_ONLY
from typing import Any, Union
from nano_agent.types import UsageMetadata


@dataclass
class AgentEvent:
    """
    Base class for all agent lifecycle events.

    Provides common metadata across all emitted events:
    - `event_type`: Discriminator string for serialization, logging, and routing.
    - `timestamp`: Epoch timestamp (in seconds) recording when the event occurred.
    """

    event_type: str = field(init=False, default="agent_event")
    _: KW_ONLY
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert the event and its nested dataclasses into a serializable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Session / Agent Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass
class AgentStartEvent(AgentEvent):
    """Emitted when the agent initiates a new run from a user prompt."""

    prompt: str
    event_type: str = field(init=False, default="agent_start")


@dataclass
class AgentEndEvent(AgentEvent):
    """Emitted when the agent loop concludes (either successfully or upon hitting limits)."""

    reply_text: str
    total_usage: UsageMetadata | None = None
    error: str | None = None
    event_type: str = field(init=False, default="agent_end")


# ---------------------------------------------------------------------------
# Turn / Step Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass
class TurnStartEvent(AgentEvent):
    """Emitted at the beginning of each reasoning step (turn) within the ReAct loop."""

    turn_index: int
    event_type: str = field(init=False, default="turn_start")


@dataclass
class TurnEndEvent(AgentEvent):
    """Emitted at the completion of a reasoning step with the step's token analytics."""

    turn_index: int
    usage: UsageMetadata | None = None
    event_type: str = field(init=False, default="turn_end")


@dataclass
class ModelFallbackEvent(AgentEvent):
    """Emitted when the primary model fails and execution fails over to the fallback model."""

    primary_model: str
    fallback_model: str
    reason: str
    event_type: str = field(init=False, default="model_fallback")


# ---------------------------------------------------------------------------
# Streaming / Token Delta Events
# ---------------------------------------------------------------------------

@dataclass
class ThinkingDeltaEvent(AgentEvent):
    """Emitted during streaming when reasoning/thinking tokens are generated (e.g. Gemini Thinking, o-series)."""

    delta: str
    event_type: str = field(init=False, default="thinking_delta")


@dataclass
class TokenDeltaEvent(AgentEvent):
    """Emitted during streaming as user-facing text tokens arrive in real-time."""

    delta: str
    event_type: str = field(init=False, default="token_delta")


# ---------------------------------------------------------------------------
# Tool Execution Lifecycle Events
# ---------------------------------------------------------------------------

@dataclass
class ToolCallStartEvent(AgentEvent):
    """Emitted immediately prior to executing an invoked tool."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    event_type: str = field(init=False, default="tool_call_start")


@dataclass
class ToolOutputDeltaEvent(AgentEvent):
    """Emitted during streaming tool execution (e.g., live stdout chunks from long-running shell commands)."""

    call_id: str
    delta: str
    event_type: str = field(init=False, default="tool_output_delta")


@dataclass
class ToolCallResultEvent(AgentEvent):
    """Emitted after a tool has completed execution, capturing output or errors."""

    call_id: str
    tool_name: str
    output: str
    is_error: bool = False
    event_type: str = field(init=False, default="tool_call_result")


# ---------------------------------------------------------------------------
# Error Handling Events
# ---------------------------------------------------------------------------

@dataclass
class ErrorEvent(AgentEvent):
    """Emitted when an unrecoverable exception or operational failure occurs."""

    message: str
    code: str | None = None
    event_type: str = field(init=False, default="error")


# Type alias for pattern matching and type hinting
AnyAgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    ModelFallbackEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallStartEvent,
    ToolOutputDeltaEvent,
    ToolCallResultEvent,
    ErrorEvent,
]

__all__ = [
    "AgentEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "ModelFallbackEvent",
    "ThinkingDeltaEvent",
    "TokenDeltaEvent",
    "ToolCallStartEvent",
    "ToolOutputDeltaEvent",
    "ToolCallResultEvent",
    "ErrorEvent",
    "AnyAgentEvent",
]