from openai import APIError, APIConnectionError
from nano_agent.config import ModelConfig, create_client, MAX_ITERATIONS
from nano_agent.memory import ConversationMemory
from nano_agent.tools import TOOLS, execute_tool
from nano_agent.types import UsageMetadata
from nano_agent.prompts import load_system_prompt


class Agent:
    """
    Autonomous ReAct (Reasoning + Acting) Agent.
    Manages dual providers (cloud + local fallback), routing,
    multi-turn conversation memory, and the internal tool execution loop.
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

    def get_model(self) -> str:
        """Returns the name of the currently active primary model."""
        if self.local and self.local_config:
            return self.local_config.model
        return self.cloud_config.model

    def _create_completion(self, messages: list[dict], tools: list[dict]):
        """
        Executes a completion call with automatic fallback between cloud and local.
        Routes to local or cloud based on `self.local`.
        """
        # Determine primary and fallback based on the `local` routing flag
        if self.local and self.local_client and self.local_config:
            primary_client, primary_config = self.local_client, self.local_config
            fallback_client, fallback_config = self.cloud_client, self.cloud_config
        else:
            primary_client, primary_config = self.cloud_client, self.cloud_config
            fallback_client, fallback_config = self.local_client, self.local_config

        try:
            return primary_client.chat.completions.create(
                model=primary_config.model,
                messages=messages,
                tools=tools,
            )
        except (APIError, APIConnectionError) as e:
            if fallback_client and fallback_config:
                if self.verbose:
                    error_msg = getattr(e, "message", str(e))
                    print(f"\n⚠️ Primary model ({primary_config.model}) unavailable: {error_msg}")
                    print(f"🔄 Switching to fallback model: {fallback_config.model}...\n")

                return fallback_client.chat.completions.create(
                    model=fallback_config.model,
                    messages=messages,
                    tools=tools,
                )
            raise e

    def run(self, user_prompt: str) -> str:
        """
        Execute an autonomous ReAct loop for a single user turn:
        1. Records user prompt to memory.
        2. Loops up to `max_steps`:
           - Calls model with current context and tools.
           - If tools requested: executes each tool, stores outputs, and continues thinking.
           - If no tools: stores final answer and returns it.
        """
        self.memory.add_user_message(user_prompt)

        for step in range(self.max_steps):
            response = self._create_completion(
                messages=self.memory.get_messages(),
                tools=TOOLS,
            )

            # Track tokens across all reasoning steps
            step_usage = UsageMetadata.from_response(response)
            self.session_usage.add(step_usage)
            if self.verbose:
                print(f"{step_usage}\n")

            message = response.choices[0].message

            # Handle Tool Calls (Action phase)
            if message.tool_calls:
                if self.verbose:
                    print("🔧 Model requested tool execution:")

                # Record the assistant's tool_calls request in memory
                self.memory.add_assistant_message(
                    content=message.content,
                    tool_calls=message.tool_calls,
                )

                # Execute all requested tools and feed back results
                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args_json = tool_call.function.arguments

                    if self.verbose:
                        print(f"   ▶ {name}({args_json})")

                    tool_result = execute_tool(name, args_json)

                    if self.verbose:
                        print(f"   ✔ Output: {tool_result}")

                    self.memory.add_tool_message(
                        tool_call_id=tool_call.id,
                        content=tool_result,
                    )

                # Continue inner loop so model can review results
                continue

            # Handle Final Text Output (No more tools needed)
            reply_text = message.content or ""
            self.memory.add_assistant_message(content=reply_text)
            return reply_text

        return f"Warning: Agent reached maximum limit of {self.max_steps} reasoning steps."

    def reset(self) -> None:
        """Reset conversation memory to start fresh."""
        self.memory.clear()