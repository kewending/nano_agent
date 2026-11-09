import json

from nano_agent.events import ApprovalRequestEvent
from nano_agent.permissions import ApprovalAnswer, RiskLevel

# Long file contents are cut to this many characters in the prompt.
PREVIEW_CHARS = 800

# Accepted inputs (after strip + lower). Anything else counts as NO.
ANSWERS: dict[str, ApprovalAnswer] = {
    "y": ApprovalAnswer.YES,
    "yes": ApprovalAnswer.YES,
    "n": ApprovalAnswer.NO,
    "no": ApprovalAnswer.NO,
    "a": ApprovalAnswer.ALWAYS,
    "always": ApprovalAnswer.ALWAYS,
}


def parse_answer(text: str) -> ApprovalAnswer | None:
    """Turn what the user typed into an answer.

    Case and surrounding spaces are ignored. Returns None for anything not in ANSWERS
    (including an empty line), so the caller can say "treated as no".
    """
    return ANSWERS.get(text.strip().lower())


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """The text itself if short, otherwise its first `limit` characters plus a note of how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more characters not shown]"


def format_approval_prompt(event: ApprovalRequestEvent) -> str:
    """Everything the user needs to see before deciding, as one block of text."""
    lines = [
        f"⚠️  {event.tool_name} needs your approval ({event.risk})",
        f"   Why: {event.reason}",
    ]
    args = event.arguments

    match event.tool_name:
        case "execute_command":
            lines.append(f"   Command: {args.get('command')}")

        case "write_file":
            lines.append(f"   File: {args.get('filepath')}")
            lines.append(f"   Content:\n{preview(str(args.get('content', '')))}")

        case "edit_file":
            lines.append(f"   File: {args.get('filepath')}")
            lines.append(f"   Replace:\n{preview(str(args.get('target_content', '')))}")
            lines.append(f"   With:\n{preview(str(args.get('replacement_content', '')))}")

        case _:
            lines.append(f"   Arguments: {json.dumps(args, indent=2)}")

    if event.risk is RiskLevel.DANGEROUS:
        lines.append("   Note: 'always' allows ANY use of this tool for the rest of the session.")
    return "\n".join(lines)


def ask_user(event: ApprovalRequestEvent) -> ApprovalAnswer:
    """CLI approver: show the request, read y/n/a. Anything unclear, Ctrl+C or Ctrl+D means NO."""
    print()
    print(format_approval_prompt(event))

    try:
        user_input = input("   Allow? [y]es / [n]o / [a]lways: ")
    except (KeyboardInterrupt, EOFError):
        print("\n   Interrupted: treated as no.")
        return ApprovalAnswer.NO

    answer = parse_answer(user_input)
    if answer is None:
        print("   Treated as no.")
        return ApprovalAnswer.NO
    
    return answer