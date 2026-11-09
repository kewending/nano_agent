import pytest
from nano_agent import agent as agent_module
from nano_agent.agent import DENIAL_PREFIX, Agent, format_denial
from nano_agent.config import ModelConfig
from nano_agent.events import ApprovalRequestEvent, ToolCallDeniedEvent, ToolCallResultEvent
from nano_agent.permissions import ApprovalAnswer, PermissionMode


@pytest.fixture
def executed(monkeypatch):
    """Replace the real execute_tool with a fake that records calls instead of running them."""
    calls = []

    def fake_execute_tool(name, arguments_json):
        calls.append(name)
        return f"fake result from {name}", False

    monkeypatch.setattr(agent_module, "execute_tool", fake_execute_tool)
    return calls


def make_agent(mode=PermissionMode.DEFAULT):
    """An Agent that never touches the network: the client is built but never called."""
    config = ModelConfig(api_key="test", base_url="http://localhost:1", model="fake", context_windows=1000)
    return Agent(cloud_config=config, verbose=False, permission_mode=mode)


def tool_call(name, arguments_json="{}", call_id="call_1"):
    """One tool call in the same shape _stream_response builds."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments_json}}


def run_calls(agent, calls, answer=None):
    """Drive _execute_tool_calls like a frontend: answer every approval request, collect all events."""
    events = []
    for event in agent._execute_tool_calls(calls):
        if isinstance(event, ApprovalRequestEvent):
            event.answer = answer
        events.append(event)
    return events

def of_type(events, cls):
    """Only the events of one class, e.g. of_type(events, ApprovalRequestEvent)."""
    return [e for e in events if isinstance(e, cls)]


# ---------------------------------------------------------------------------
# The permission gate in _execute_tool_calls
# ---------------------------------------------------------------------------

def test_safe_tool_runs_without_asking(executed):
    """A SAFE tool (read_file) runs straight away."""
    events = run_calls(make_agent(), [tool_call("read_file")])

    assert executed == ["read_file"]
    assert of_type(events, ApprovalRequestEvent) == []


def test_mutating_tool_asks_first(executed):
    """A MUTATING tool (write_file) yields exactly one ApprovalRequestEvent before running."""
    events = run_calls(make_agent(), [tool_call("write_file")], answer=ApprovalAnswer.YES)
    approvals = of_type(events, ApprovalRequestEvent)

    assert len(approvals) == 1
    assert approvals[0].tool_name == "write_file"


def test_yes_runs_the_tool(executed):
    """Answering YES lets the tool run."""
    run_calls(make_agent(), [tool_call("write_file")], answer=ApprovalAnswer.YES)

    assert executed == ["write_file"]


def test_no_blocks_the_tool(executed):
    """Answering NO blocks the tool and says so with a ToolCallDeniedEvent."""
    events = run_calls(make_agent(), [tool_call("write_file")], answer=ApprovalAnswer.NO)

    assert executed == []
    assert len(of_type(events, ToolCallDeniedEvent)) == 1


def test_no_answer_blocks_the_tool(executed):
    """Fail closed: a frontend that never sets event.answer must not run the tool."""
    run_calls(make_agent(), [tool_call("write_file")])

    assert executed == []


def test_always_stops_asking_next_time(executed):
    """ALWAYS runs the tool now and grants it, so the next call runs without asking.

    The second run answers None on purpose: if the grant didn't work, None would deny.
    """
    agent = make_agent()

    run_calls(agent, [tool_call("write_file")], answer=ApprovalAnswer.ALWAYS)
    second_run = run_calls(agent, [tool_call("write_file")])

    assert executed == ["write_file", "write_file"]
    assert of_type(second_run, ApprovalRequestEvent) == []


def test_read_only_mode_blocks_without_asking(executed):
    """In READ_ONLY mode a MUTATING tool is denied without a question."""
    events = run_calls(make_agent(PermissionMode.READ_ONLY), [tool_call("write_file")])

    assert executed == []
    assert len(of_type(events, ToolCallDeniedEvent)) == 1
    assert of_type(events, ApprovalRequestEvent) == []


def test_hard_deny_blocks_without_asking(executed):
    """A hard-denied command is blocked even if the user would say YES.

    Also guards the `is not dict` bug: the policy must see the real "command".
    """
    call = tool_call("execute_command", arguments_json='{"command": "rm -rf /"}')
    events = run_calls(make_agent(), [call], answer=ApprovalAnswer.YES)
    denials = of_type(events, ToolCallDeniedEvent)

    assert executed == []
    assert of_type(events, ApprovalRequestEvent) == []
    assert len(denials) == 1
    assert "hard security" in denials[0].reason


def test_non_dict_arguments_do_not_crash(executed):
    """Arguments that parse to a non-dict (e.g. "5") are wrapped, not crashed on."""
    call = tool_call("execute_command", arguments_json="5")
    events = run_calls(make_agent(), [call], answer=ApprovalAnswer.NO)

    assert executed == []
    assert len(of_type(events, ToolCallDeniedEvent)) == 1


def tool_messages(agent):
    """The tool messages in the agent's memory, in order."""
    return [m for m in agent.memory.get_messages() if m["role"] == "tool"]


# ---------------------------------------------------------------------------
# Denied calls still answer the model
# ---------------------------------------------------------------------------

def test_format_denial_includes_prefix_and_reason():
    """format_denial starts with DENIAL_PREFIX and contains the policy's reason word for word."""
    message = format_denial("write_file denied by you")

    assert message.startswith(DENIAL_PREFIX)
    assert "write_file denied by you" in message


def test_format_denial_tells_model_not_to_retry():
    """The message tells the model NOT to retry, so it adapts instead of looping."""
    message = format_denial("write_file denied by you")

    assert "do not retry" in message.lower()


def test_denied_call_still_gets_a_tool_message(executed):
    """A call the user denies still gets a tool message with the same tool_call_id."""
    agent = make_agent()

    run_calls(agent, [tool_call("write_file", call_id="call_1")], answer=ApprovalAnswer.NO)
    messages = tool_messages(agent)

    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "call_1"
    assert messages[0]["content"].startswith(DENIAL_PREFIX)


def test_every_tool_call_gets_exactly_one_tool_message(executed):
    """Mixed batch: an allowed call and a denied call both get answered, in order.

    This is the exact rule the API enforces (otherwise 400).
    """
    agent = make_agent()
    calls = [
        tool_call("read_file", call_id="call_1"),
        tool_call("write_file", call_id="call_2"),
    ]

    run_calls(agent, calls, answer=ApprovalAnswer.NO)

    assert [m["tool_call_id"] for m in tool_messages(agent)] == ["call_1", "call_2"]
    assert executed == ["read_file"]


def test_denied_call_yields_no_result_event(executed):
    """A denial is reported once, as ToolCallDeniedEvent, not also as a ToolCallResultEvent."""
    events = run_calls(make_agent(), [tool_call("write_file")], answer=ApprovalAnswer.NO)

    assert of_type(events, ToolCallResultEvent) == []
    assert len(of_type(events, ToolCallDeniedEvent)) == 1
