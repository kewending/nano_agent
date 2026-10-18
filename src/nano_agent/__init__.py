import sys
from nano_agent.agent import Agent
from nano_agent.config import get_model_config

# Ensure Windows terminal prints emojis and special characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    # Set up dual configurations: Cloud (Gemini) and Local (Llama)
    cloud_config = get_model_config("gemini")
    local_config = get_model_config("llama")

    # Set local=True to use local model primarily (with cloud fallback),
    # or local=False to use cloud primarily (with local fallback).
    agent = Agent(
        cloud_config=cloud_config,
        local_config=local_config,
        local=True,
        verbose=True,
    )

    print(f"🤖 Connected to {agent.get_model()}. Type 'exit', 'quit', or 'q' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye!")
            break

        response = agent.run(user_input)
        print(f"Assistant: {response}\n")

    # Print total cumulative tokens consumed during the entire session
    print("-" * 50)
    print(f"📈 Total Session Usage: {agent.session_usage}")