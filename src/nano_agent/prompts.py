from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_system_prompt(filename: str = "system.md") -> str:
    """Load a markdown prompt file from the prompts directory."""
    prompt_file = PROMPTS_DIR / filename
    if not prompt_file.exists():
        return "You are a helpful assistant."  # fallback default
        
    return prompt_file.read_text(encoding="utf-8").strip()