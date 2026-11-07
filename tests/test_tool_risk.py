import pytest
from nano_agent import tools
from nano_agent.permissions import RiskLevel


@pytest.fixture
def empty_registries(monkeypatch):
    """Fresh, empty tool registries for one test; restored automatically after."""
    monkeypatch.setattr(tools, "TOOLS", [])
    monkeypatch.setattr(tools, "TOOL_REGISTRY", {})
    monkeypatch.setattr(tools, "TOOL_METADATA", {})


# ---------------------------------------------------------------------------
# The real tool registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("tool_name", "expected_risk"),
    [
        ("get_current_time", RiskLevel.SAFE),
        ("calculate", RiskLevel.SAFE),
        ("read_file", RiskLevel.SAFE),
        ("list_directory", RiskLevel.SAFE),
        ("write_file", RiskLevel.MUTATING),
        ("edit_file", RiskLevel.MUTATING),
        ("execute_command", RiskLevel.DANGEROUS),
    ],
)
def test_tool_risk_levels(tool_name, expected_risk):
    assert tools.get_tool_risk(tool_name) == expected_risk


def test_every_real_tool_has_metadata():
    assert set(tools.TOOL_REGISTRY) == set(tools.TOOL_METADATA)


def test_unknown_tool_is_dangerous():
    assert tools.get_tool_risk("not_a_real_tool") == RiskLevel.DANGEROUS


def test_risk_not_sent_to_model():
    for schema in tools.TOOLS:
        assert set(schema["function"]) == {"name", "description", "parameters"}


# ---------------------------------------------------------------------------
# The @tool decorator
# ---------------------------------------------------------------------------

def test_decorator_fills_every_registry(empty_registries):
    @tools.tool(risk=RiskLevel.SAFE)
    def dummy():
        """dummy function"""

    assert "dummy" in tools.TOOL_REGISTRY
    assert "dummy" in tools.TOOL_METADATA
    assert len(tools.TOOLS) == 1


def test_bare_decorator_defaults_to_dangerous(empty_registries):
    @tools.tool
    def dummy():
        """dummy function"""

    assert tools.TOOL_METADATA["dummy"].risk == RiskLevel.DANGEROUS


def test_empty_parentheses_default_to_dangerous(empty_registries):
    @tools.tool()
    def dummy():
        """dummy function"""

    assert tools.TOOL_METADATA["dummy"].risk == RiskLevel.DANGEROUS


def test_decorated_function_still_callable(empty_registries):
    @tools.tool(risk=RiskLevel.SAFE)
    def dummy():
        """dummy function"""
        return "ok"

    assert dummy() == "ok"


def test_registering_same_name_twice_raises(empty_registries):
    @tools.tool(risk=RiskLevel.SAFE)
    def dummy():
        """first registration"""

    with pytest.raises(ValueError, match="already registered"):

        @tools.tool(risk=RiskLevel.SAFE)
        def dummy():  # noqa: F811  (redefinition is the point of this test)
            """second registration with the same name"""


def test_string_risk_raises_type_error(empty_registries):
    with pytest.raises(TypeError, match="got str: 'safe'"):
        tools.tool(risk="safe")


def test_positional_risk_gives_clear_error(empty_registries):
    with pytest.raises(TypeError, match="risk="):
        tools.tool(RiskLevel.SAFE)