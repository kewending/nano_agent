import pytest
from nano_agent.approval import ask_user, format_approval_prompt, parse_answer, preview
from nano_agent.events import ApprovalRequestEvent
from nano_agent.permissions import ApprovalAnswer, RiskLevel


def make_event(tool_name="write_file", risk=RiskLevel.MUTATING, arguments=None):
    """An ApprovalRequestEvent with sensible defaults."""
    return ApprovalRequestEvent(
        call_id="call_1",
        tool_name=tool_name,
        risk=risk,
        reason=f"{tool_name} needs your approval",
        arguments={} if arguments is None else arguments,
    )


def fake_input(monkeypatch, result):
    """Make input() return `result`, or raise it if it is an exception class."""
    def _input(prompt=""):
        if isinstance(result, type) and issubclass(result, BaseException):
            raise result
        return result

    monkeypatch.setattr("builtins.input", _input)


# ---------------------------------------------------------------------------
# parse_answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("y", ApprovalAnswer.YES),
        ("yes", ApprovalAnswer.YES),
        ("Y", ApprovalAnswer.YES),
        (" yes ", ApprovalAnswer.YES),
        ("n", ApprovalAnswer.NO),
        ("no", ApprovalAnswer.NO),
        ("a", ApprovalAnswer.ALWAYS),
        ("ALWAYS", ApprovalAnswer.ALWAYS),
        # Must NOT be accepted:
        ("", None),
        ("x", None),
        ("yes please", None),
        ("ye", None),
    ],
)
def test_parse_answer(text, expected):
    """Every accepted spelling maps to its answer; anything else gives None."""
    assert parse_answer(text) is expected


# ---------------------------------------------------------------------------
# preview / format_approval_prompt
# ---------------------------------------------------------------------------

def test_preview_keeps_short_text():
    """Text within the limit comes back unchanged."""
    assert preview("hello", limit=10) == "hello"
    assert preview("x" * 10, limit=10) == "x" * 10  # exactly at the limit: not cut


def test_preview_cuts_long_text_and_says_so():
    """Longer text is cut to the limit and mentions how many characters were hidden."""
    result = preview("x" * 30, limit=10)

    assert result.startswith("x" * 10)
    assert "x" * 11 not in result
    assert "20 more characters" in result


def test_prompt_shows_the_command():
    """For execute_command, the exact command is in the prompt."""
    event = make_event("execute_command", RiskLevel.DANGEROUS, {"command": "pip install requests"})

    assert "pip install requests" in format_approval_prompt(event)


def test_prompt_shows_file_path_and_content():
    """For write_file, both the filepath and the content are in the prompt."""
    event = make_event(arguments={"filepath": "notes/hello.txt", "content": "hi there"})

    prompt = format_approval_prompt(event)

    assert "notes/hello.txt" in prompt
    assert "hi there" in prompt


def test_prompt_shows_edit_target_and_replacement():
    """For edit_file, the user sees what is replaced and what replaces it."""
    event = make_event("edit_file", arguments={
        "filepath": "app.py", "target_content": "old_line", "replacement_content": "new_line",
    })

    prompt = format_approval_prompt(event)

    assert "app.py" in prompt
    assert "old_line" in prompt
    assert "new_line" in prompt


def test_prompt_cuts_long_content():
    """A huge file body is previewed, not dumped into the terminal."""
    event = make_event(arguments={"filepath": "big.txt", "content": "x" * 5000})

    prompt = format_approval_prompt(event)

    assert "x" * 5000 not in prompt
    assert "more characters not shown" in prompt


def test_prompt_falls_back_to_json_for_other_tools():
    """Unknown tools still show all their arguments."""
    event = make_event("mystery_tool", arguments={"target": "db", "level": 3})

    prompt = format_approval_prompt(event)

    assert '"target": "db"' in prompt
    assert '"level": 3' in prompt


def test_prompt_survives_missing_arguments():
    """A model call with missing arguments must not crash the prompt."""
    for tool in ("execute_command", "write_file", "edit_file"):
        format_approval_prompt(make_event(tool, arguments={}))  # no exception


def test_prompt_shows_reason_and_risk():
    """The reason and the risk level are always shown."""
    prompt = format_approval_prompt(make_event("write_file", RiskLevel.MUTATING))

    assert "write_file needs your approval" in prompt
    assert str(RiskLevel.MUTATING) in prompt


def test_prompt_warns_always_for_dangerous_tools():
    """DANGEROUS tools warn that 'always' allows ANY use; MUTATING tools don't."""
    dangerous = format_approval_prompt(make_event("execute_command", RiskLevel.DANGEROUS))
    mutating = format_approval_prompt(make_event("write_file", RiskLevel.MUTATING))

    assert "ANY" in dangerous
    assert "ANY" not in mutating


# ---------------------------------------------------------------------------
# ask_user (input() is faked with monkeypatch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("y", ApprovalAnswer.YES),
        ("a", ApprovalAnswer.ALWAYS),
        ("n", ApprovalAnswer.NO),
        ("", ApprovalAnswer.NO),
        ("whatever", ApprovalAnswer.NO),
        (" Y ", ApprovalAnswer.YES),
    ],
)
def test_ask_user_returns_the_answer(monkeypatch, typed, expected):
    """What the user types becomes the answer; empty or unknown input is NO."""
    fake_input(monkeypatch, typed)

    assert ask_user(make_event()) is expected


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_ask_user_ctrl_c_or_ctrl_d_means_no(monkeypatch, error):
    """Ctrl+C / Ctrl+D while asking returns NO instead of crashing the session."""
    fake_input(monkeypatch, error)

    assert ask_user(make_event()) is ApprovalAnswer.NO


def test_ask_user_shows_the_prompt_before_asking(monkeypatch, capsys):
    """The user sees what they are approving (the file path) before typing."""
    fake_input(monkeypatch, "n")

    ask_user(make_event(arguments={"filepath": "secret_plan.txt", "content": "..."}))

    assert "secret_plan.txt" in capsys.readouterr().out


def test_ask_user_says_when_input_is_treated_as_no(monkeypatch, capsys):
    """Unknown input prints "Treated as no" so a typo isn't a silent surprise."""
    fake_input(monkeypatch, "yse")

    ask_user(make_event())

    assert "treated as no" in capsys.readouterr().out.lower()


def test_ask_user_is_quiet_on_a_clear_answer(monkeypatch, capsys):
    """A valid answer does NOT print the 'treated as no' warning."""
    fake_input(monkeypatch, "y")

    ask_user(make_event())

    assert "treated as no" not in capsys.readouterr().out.lower()
