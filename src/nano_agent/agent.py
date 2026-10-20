import json
from collections.abc import Generator
from typing import Any
from openai import APIConnectionError, APIError

from nano_agent.config import MAX_ITERATIONS, ModelConfig, create_client
from nano_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    ErrorEvent,
    ModelFallbackEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from nano_agent.memory import ConversationMemory
from nano_agent.prompts import load_system_prompt
from nano_agent.tools import TOOLS, execute_tool
from nano_agent.types import UsageMetadata


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

    def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> Generator[AgentEvent, None, None]:
        """
        Executes a list of tool calls:
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

            yield ToolCallStartEvent(
                call_id=call_id,
                tool_name=tool_name,
                arguments=parsed_args,
            )

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

    def run(self, user_prompt: str) -> str:
        """
        Synchronous execution wrapper that runs the event stream.
        Maintains backward compatibility while completely decoupling
        the ReAct loop logic from console output.
        """
        final_reply = ""
        for event in self.run_stream(user_prompt):
            if self.verbose:
                self._render_event(event)
            if isinstance(event, AgentEndEvent):
                final_reply = event.reply_text
        return final_reply

    def reset(self) -> None:
        """Reset conversation memory to start fresh."""
        self.memory.clear()