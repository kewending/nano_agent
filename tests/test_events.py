import pytest
from nano_agent.events import ApprovalRequestEvent, ToolCallDeniedEvent
from nano_agent.permissions import ApprovalAnswer, RiskLevel


def make_approval_event(**overrides):
    """Build an ApprovalRequestEvent with sensible defaults; override any field by keyword."""
    fields = {
        "call_id": "call_1",
        "tool_name": "write_file",
        "risk": RiskLevel.MUTATING,
        "reason": "write_file can change files and needs your approval",
    }
    fields.update(overrides)
    return ApprovalRequestEvent(**fields)


# ---------------------------------------------------------------------------
# ApprovalAnswer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (ApprovalAnswer.YES, "yes"),
        (ApprovalAnswer.NO, "no"),
        (ApprovalAnswer.ALWAYS, "always"),
    ],
)
def test_approval_answer_values(answer, expected):
    assert answer == expected


def test_unknown_approval_answer_raises():
    with pytest.raises(ValueError, match="maybe"):
        ApprovalAnswer("maybe")


# ---------------------------------------------------------------------------
# ApprovalRequestEvent
# ---------------------------------------------------------------------------

def test_approval_request_starts_unanswered():
    event = make_approval_event()

    assert event.answer is None


def test_approval_request_answer_can_be_set():
    event = make_approval_event()

    event.answer = ApprovalAnswer.ALWAYS

    assert event.answer is ApprovalAnswer.ALWAYS


def test_approval_request_event_type():
    assert make_approval_event().event_type == "approval_request"


def test_approval_request_arguments_not_shared():
    first = make_approval_event()
    second = make_approval_event()

    first.arguments["filepath"] = "a.txt"

    assert second.arguments == {}


def test_approval_request_to_dict():
    event = make_approval_event(arguments={"filepath": "a.txt"})

    data = event.to_dict()

    assert data["answer"] is None
    assert data["risk"] == "mutating"
    assert data["reason"] == event.reason
    assert data["arguments"] == {"filepath": "a.txt"}
    assert data["event_type"] == "approval_request"


def test_answer_seen_by_generator_after_yield():
    """The decision slot: the consumer fills it, the generator reads it on resume."""
    seen = []

    def agent_like():
        event = make_approval_event()
        yield event
        seen.append(event.answer)

    for event in agent_like():
        event.answer = ApprovalAnswer.YES

    assert seen == [ApprovalAnswer.YES]


def test_unanswered_request_stays_none_after_yield():
    """A consumer that ignores the event leaves the answer as None, which means 'no'."""
    seen = []

    def agent_like():
        event = make_approval_event()
        yield event
        seen.append(event.answer)

    for _ in agent_like():
        pass

    assert seen == [None]


# ---------------------------------------------------------------------------
# ToolCallDeniedEvent
# ---------------------------------------------------------------------------

def test_tool_call_denied_event_type():
    event = ToolCallDeniedEvent(call_id="call_1", tool_name="execute_command", reason="blocked")

    assert event.event_type == "tool_call_denied"
    assert event.reason == "blocked"
