import json
from collections.abc import Callable, Generator
from typing import Any
from openai import APIConnectionError, APIError

from nano_agent.config import MAX_ITERATIONS, ModelConfig, create_client
from nano_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ApprovalRequestEvent,
    ErrorEvent,
    ModelFallbackEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallResultEvent,
    ToolCallDeniedEvent,
    ToolCallStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from nano_agent.memory import ConversationMemory
from nano_agent.permissions import (
    ApprovalAnswer,
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    Verdict,
)
from nano_agent.prompts import load_system_prompt
from nano_agent.tools import TOOLS, execute_tool, get_tool_risk
from nano_agent.types import UsageMetadata, ContextBreakdown
from nano_agent.tokens import count_tokens, count_payload_tokens


DENIAL_PREFIX = "Permission denied:"

# A frontend function that answers one approval request (CLI prompt, web UI, test fake).
# Returning None means "no answer", which denies the call.
Approver = Callable[[ApprovalRequestEvent], ApprovalAnswer | None]


def format_denial(reason: str) -> str:
    """The tool message the model reads when a call is blocked.

    Says why, and tells the model not to retry the same call, so it adapts instead of looping.
    """
    return f"{DENIAL_PREFIX} {reason}. Do not retry this call; try a different approach or ask the user."


class Agent:
    """
    Autonomous ReAct (Reasoning + Acting) Agent.
    Manages dual providers (cloud + local fallback), routing,
    multi-turn conversation memory, and the real-time event-driven execution loop.
    """

    def __init__(
        self,
        cloud_config: ModelConfig,
        local_config: ModelConfig | None = None,
        verbose: bool = True,
        local: bool = False,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self.cloud_config = cloud_config
        self.cloud_client = create_client(cloud_config)

        self.local_config = local_config
        self.local_client = create_client(local_config) if local_config else None

        self.local = local
        self.system_prompt = load_system_prompt()
        self.memory = ConversationMemory(system_prompt=self.system_prompt)
        self.session_usage = UsageMetadata()
        self.max_steps = MAX_ITERATIONS
        self.verbose = verbose
        self._is_thinking = False
        self.policy = PermissionPolicy(mode=permission_mode)

    def get_model(self) -> str:
        """Returns the name of the currently active primary model."""
        if self.local and self.local_config:
            return self.local_config.model
        return self.cloud_config.model

    def _create_completion(
        self, messages: list[dict], tools: list[dict], stream: bool = True
    ) -> tuple[Any, ModelFallbackEvent | None]:
        """
        Executes a completion call with automatic fallback between cloud and local.
        Routes to local or cloud based on `self.local`.
        Returns a tuple of (completion_stream, optional_fallback_event).
        """
        if self.local and self.local_client and self.local_config:
            primary_client, primary_config = self.local_client, self.local_config
            fallback_client, fallback_config = self.cloud_client, self.cloud_config
        else:
            primary_client, primary_config = self.cloud_client, self.cloud_config
            fallback_client, fallback_config = self.local_client, self.local_config

        fallback_event: ModelFallbackEvent | None = None

        try:
            response = primary_client.chat.completions.create(
                model=primary_config.model,
                messages=messages,
                tools=tools,
                stream=stream,
                stream_options={"include_usage": True} if stream else None,
            )
            return response, None
        except (APIError, APIConnectionError) as e:
            if fallback_client and fallback_config:
                error_msg = getattr(e, "message", str(e))
                fallback_event = ModelFallbackEvent(
                    primary_model=primary_config.model,
                    fallback_model=fallback_config.model,
                    reason=error_msg,
                )
                response = fallback_client.chat.completions.create(
                    model=fallback_config.model,
                    messages=messages,
                    tools=tools,
                    stream=stream,
                    stream_options={"include_usage": True} if stream else None,
                )
                return response, fallback_event
            raise e

    def _stream_response(
        self, response: Any
    ) -> Generator[AgentEvent, None, tuple[str, list[dict[str, Any]], UsageMetadata]]:
        """
        Consumes chunks from the streaming LLM response:
        - Yields real-time ThinkingDeltaEvent and TokenDeltaEvent.
        - Accumulates fragmented tool call chunks.
        - Captures token usage from the final chunk.
        Returns (accumulated_content, reconstructed_tool_calls, step_usage).
        """
        accumulated_content = ""
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        step_usage = UsageMetadata()

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield ThinkingDeltaEvent(delta=reasoning)

                if delta.content:
                    accumulated_content += delta.content
                    yield TokenDeltaEvent(delta=delta.content)

                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_chunk.id or "",
                                "name": tc_chunk.function.name if tc_chunk.function else "",
                                "arguments": [],
                            }
                        if tc_chunk.id:
                            tool_calls_acc[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name:
                            tool_calls_acc[idx]["name"] = tc_chunk.function.name
                        if tc_chunk.function and tc_chunk.function.arguments:
                            tool_calls_acc[idx]["arguments"].append(tc_chunk.function.arguments)

            if chunk.usage:
                step_usage = UsageMetadata.from_response(chunk)

        reconstructed_tool_calls = [
            {
                "id": v["id"],
                "type": "function",
                "function": {
                    "name": v["name"],
                    "arguments": "".join(v["arguments"]),
                },
            }
            for v in tool_calls_acc.values()
        ]

        return accumulated_content, reconstructed_tool_calls, step_usage

    def _check_permission(
        self, call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> Generator[AgentEvent, None, PermissionDecision]:
        """
        Asks the policy whether one tool call may run.
        - ALLOW / DENY: returned straight away, nothing is yielded.
        - ASK: yields an ApprovalRequestEvent, then reads the answer the frontend
          wrote into it. Anything other than YES or ALWAYS (including no answer) denies.
        Always returns ALLOW or DENY, never ASK.
        """
        risk = get_tool_risk(tool_name)
        request = PermissionRequest(tool_name=tool_name, risk=risk, arguments=arguments)
        decision = self.policy.evaluate(request)

        if decision.verdict is not Verdict.ASK:
            return decision

        event = ApprovalRequestEvent(
            call_id=call_id,
            tool_name=tool_name,
            risk=risk,
            reason=decision.reason,
            arguments=arguments,
        )
        yield event  # pauses here; the frontend sets event.answer before we resume

        match event.answer:
            case ApprovalAnswer.ALWAYS:
                self.policy.grant(request)
                return PermissionDecision(
                    verdict=Verdict.ALLOW,
                    reason=f"{tool_name} approved by you for this session",
                )
            case ApprovalAnswer.YES:
                return PermissionDecision(
                    verdict=Verdict.ALLOW,
                    reason=f"{tool_name} approved by you",
                )
            case _:
                return PermissionDecision(
                    verdict=Verdict.DENY,
                    reason=f"{tool_name} denied by you",
                )

    def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> Generator[AgentEvent, None, None]:
        """
        Executes a list of tool calls:
        - Checks each call against self.policy first (may yield ApprovalRequestEvent).
        - Denied calls yield ToolCallDeniedEvent and are not run.
        - Yields ToolCallStartEvent and ToolCallResultEvent.
        - Adds tool outputs to conversation memory.
        """
        for tool_call in tool_calls:
            call_id = tool_call["id"]
            tool_name = tool_call["function"]["name"]
            args_json = tool_call["function"]["arguments"]

            try:
                parsed_args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                parsed_args = {"raw": args_json}

            if not isinstance(parsed_args, dict):  # e.g. the model sent "5" or "[1, 2]"
                parsed_args = {"raw": args_json}

            yield ToolCallStartEvent(
                call_id=call_id,
                tool_name=tool_name,
                arguments=parsed_args,
            )

            decision = yield from self._check_permission(call_id, tool_name, parsed_args)

            if decision.verdict is Verdict.DENY:
                yield ToolCallDeniedEvent(
                    call_id=call_id,
                    tool_name=tool_name,
                    reason=decision.reason,
                )
                # Every tool_call_id must get a tool message, or the next API call fails with 400.
                self.memory.add_tool_message(
                    tool_call_id=call_id,
                    content=format_denial(decision.reason),
                )
                continue

            tool_result, is_error = execute_tool(tool_name, args_json)

            yield ToolCallResultEvent(
                call_id=call_id,
                tool_name=tool_name,
                output=tool_result,
                is_error=is_error,
            )

            self.memory.add_tool_message(
                tool_call_id=call_id,
                content=tool_result,
            )
    def get_context_breakdown(self) -> ContextBreakdown:
        """Inspects current memory and tool registry to calculate context consumption."""
        system_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        tool_calls_tokens = 0
        tool_results_tokens = 0
        # 1. Inspect conversation history
        for msg in self.memory.get_messages():
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                system_tokens += count_tokens(content)
            elif role == "user":
                user_tokens += count_tokens(content)
            elif role == "assistant":
                if content:
                    assistant_tokens += count_tokens(content)
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls_tokens += count_payload_tokens(msg["tool_calls"])
            elif role == "tool":
                tool_results_tokens += count_tokens(content)
        # 2. Tool Registry (the JSON definitions passed to tools=[...])
        tools_schema_tokens = count_payload_tokens(TOOLS)
        # 3. Context window limit from active config
        active_config = self.local_config if (self.local and self.local_config) else self.cloud_config
        context_limit = int(getattr(active_config, "context_windows"))
        total_tokens = (
            system_tokens
            + tools_schema_tokens
            + user_tokens
            + assistant_tokens
            + tool_calls_tokens
            + tool_results_tokens
        )
        return ContextBreakdown(
            system_prompt_tokens=system_tokens,
            tools_schema_tokens=tools_schema_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            tool_calls_tokens=tool_calls_tokens,
            tool_results_tokens=tool_results_tokens,
            total_tokens=total_tokens,
            context_limit=context_limit,
        )

    def run_stream(self, user_prompt: str) -> Generator[AgentEvent, None, str]:
        """
        Execute an autonomous ReAct loop as a typed real-time event stream:
        1. Records user prompt to memory and emits AgentStartEvent.
        2. Loops up to `max_steps`:
           - Emits TurnStartEvent.
           - Streams LLM response (delegated via _stream_response).
           - If tools requested: executes tools (delegated via _execute_tool_calls).
           - If final answer reached: returns answer and finishes.
        3. If max steps exceeded: emits AgentEndEvent with warning.
        """
        self.memory.add_user_message(user_prompt)
        yield AgentStartEvent(prompt=user_prompt)

        for step in range(self.max_steps):
            yield TurnStartEvent(turn_index=step)

            try:
                response, fallback_event = self._create_completion(
                    messages=self.memory.get_messages(),
                    tools=TOOLS,
                    stream=True,
                )
            except Exception as e:
                error_msg = str(e)
                yield ErrorEvent(message=error_msg, code="API_ERROR")
                yield AgentEndEvent(
                    reply_text=f"Error contacting model: {error_msg}",
                    total_usage=self.session_usage,
                    error=error_msg,
                )
                return f"Error contacting model: {error_msg}"

            if fallback_event:
                yield fallback_event

            # 1. Consume token stream & accumulate tool calls
            content, tool_calls, step_usage = yield from self._stream_response(response)
            self.session_usage.add(step_usage)

            # 2. Action phase: Execute tools if requested
            if tool_calls:
                self.memory.add_assistant_message(
                    content=content or None,
                    tool_calls=tool_calls,
                )
                yield from self._execute_tool_calls(tool_calls)
                yield TurnEndEvent(turn_index=step, usage=step_usage)
                continue

            # 3. Final answer reached
            self.memory.add_assistant_message(content=content)
            yield TurnEndEvent(turn_index=step, usage=step_usage)
            yield AgentEndEvent(
                reply_text=content,
                total_usage=self.session_usage,
            )
            return content

        # Max steps reached without final answer
        warning_msg = f"Warning: Agent reached maximum limit of {self.max_steps} reasoning steps."
        yield AgentEndEvent(
            reply_text=warning_msg,
            total_usage=self.session_usage,
            error="max_steps_exceeded",
        )
        return warning_msg

    def _render_event(self, event: AgentEvent) -> None:
        """Helper to render lifecycle events to console with clean styling."""
        match event:
            case ModelFallbackEvent(fallback_model=fb, reason=reason):
                if self._is_thinking:
                    print("\033[0m")
                    self._is_thinking = False
                print(f"\n⚠️ Primary model unavailable: {reason}")
                print(f"🔄 Switching to fallback model: {fb}...\n")

            case ThinkingDeltaEvent(delta=delta):
                if not self._is_thinking:
                    print("\n💭 Thinking:\n  \033[90m", end="", flush=True)
                    self._is_thinking = True
                print(delta, end="", flush=True)

            case TokenDeltaEvent(delta=delta):
                if self._is_thinking:
                    print("\033[0m\n\nAssistant: ", end="", flush=True)
                    self._is_thinking = False
                print(delta, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                if self._is_thinking:
                    print("\033[0m\n")
                    self._is_thinking = False
                print("\n🔧 Model requested tool execution:")
                print(f"   ▶ {name}({json.dumps(args)})")

            case ToolCallDeniedEvent(reason=reason):
                print(f"   🚫 Denied: {reason}")

            case ToolCallResultEvent(output=out, is_error=err):
                status = "❌" if err else "✔"
                print(f"   {status} Output: {out}")

            case TurnEndEvent(usage=usage) if usage and usage.total_tokens > 0:
                if self._is_thinking:
                    print("\033[0m")
                    self._is_thinking = False
                print(f"\n{usage}\n")

            case AgentEndEvent():
                if self._is_thinking:
                    print("\033[0m")
                    self._is_thinking = False
                print()

            case _:
                pass

    def run(self, user_prompt: str, approver: Approver | None = None) -> str:
        """
        Synchronous execution wrapper that runs the event stream.
        Maintains backward compatibility while completely decoupling
        the ReAct loop logic from console output.

        `approver` answers ApprovalRequestEvents. Without one, every request
        stays unanswered and is denied (fail closed).
        """
        final_reply = ""
        for event in self.run_stream(user_prompt):
            if self.verbose:
                self._render_event(event)
            # Write the answer into the event's decision slot before the generator resumes.
            if isinstance(event, ApprovalRequestEvent) and approver is not None:
                event.answer = approver(event)

            if isinstance(event, AgentEndEvent):
                final_reply = event.reply_text
        return final_reply

    def reset(self) -> None:
        """Reset conversation memory to start fresh."""
        self.memory.clear()