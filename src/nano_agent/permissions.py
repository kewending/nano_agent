from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any

SHELL_TOOL = "execute_command"
SHELL_ARG = "command"

# What `rm` / `rd` / `Remove-Item` must be pointed at to count as a wipe: the
# whole argument must be a root, home or wildcard, so `./build` or `/tmp/x` don't count.
_WIPE_TARGET = r"(/|/\*|~|~/|\*|[a-z]:\\?|\\)( |$)"

# Commands so destructive that nothing may allow them: not a grant, not
# auto-approve. These patterns run on normalised text (single spaces).
#
# This is a speed bump, not a wall. It catches obvious accidents an LLM might
# make; a determined bypass (quotes, variables, base64) gets through, which is
# why execute_command still asks every time. Keep the rules narrow: a false
# positive here can never be overridden by the user.
HARD_DENY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        # rm with a recursive flag (-r, -rf, -fr, --recursive) on / ~ or *
        re.compile(
            r"\brm( -{1,2}[a-z-]+)* (-[a-z]*r[a-z]*|--recursive)( -{1,2}[a-z-]+)* " + _WIPE_TARGET,
            re.IGNORECASE,
        ),
        "recursive delete of the root, home folder or everything here",
    ),
    (
        # PowerShell: flags and path can come in any order, so use two lookaheads
        re.compile(
            r"\b(remove-item|ri)\b(?=.* -r(ecurse)?\b)(?=.* " + _WIPE_TARGET + ")",
            re.IGNORECASE,
        ),
        "recursive delete of a drive root or home folder",
    ),
    (
        # Windows cmd: rd /s on a drive root
        re.compile(r"\b(rd|rmdir)\b(?=.* /s\b)(?=.* " + _WIPE_TARGET + ")", re.IGNORECASE),
        "recursive delete of a drive root",
    ),
    (
        # --force or a short-flag group containing f (-f, -uf); not --force-with-lease
        re.compile(r"\bgit push\b.* (--force|-[a-z]*f[a-z]*)( |$)", re.IGNORECASE),
        "force-push can overwrite shared history",
    ),
    (
        re.compile(r"\b(curl|wget)\b.*\| ?(sudo )?(sh|bash)\b", re.IGNORECASE),
        "runs a script downloaded from the internet",
    ),
    (
        re.compile(r"\b(shutdown|reboot)\b", re.IGNORECASE),
        "shuts down or restarts the machine",
    ),
)


def _normalize_command(command: str) -> str:
    """Collapse every run of whitespace to one space and trim the ends."""
    return " ".join(command.split())


class RiskLevel(StrEnum):
    """How much harm a tool can do if it runs without the user's approval."""

    SAFE = "safe"              # read-only, auto-approve
    MUTATING = "mutating"      # changes files, ask
    DANGEROUS = "dangerous"    # runs arbitrary things, ask (louder)


class Verdict(StrEnum):
    """Outcome of a permission check for one tool call, made before it runs."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionMode(StrEnum):
    """Session-wide setting that changes how strict the policy is."""

    DEFAULT = "default"
    READ_ONLY = "read_only"
    AUTO_APPROVE = "auto_approve"


class ApprovalAnswer(StrEnum):
    """The user's answer when asked to approve one tool call."""

    YES = "yes"         # run it this time
    NO = "no"           # don't run it
    ALWAYS = "always"   # run it, and stop asking for this tool this session


@dataclass(frozen=True)
class PermissionDecision:
    """The policy's answer for one tool call: a verdict and why."""

    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class PermissionRequest:
    """One tool call waiting for a permission decision.

    `risk` is supplied by the caller (via tools.get_tool_risk) so this module
    never has to import the tool registry. `arguments` is a read-only Mapping:
    the policy inspects it and must never modify it.
    """

    tool_name: str
    risk: RiskLevel
    arguments: Mapping[str, Any]


@dataclass
class PermissionPolicy:
    """Decides whether a tool call may run: allow, deny, or ask the user."""

    mode: PermissionMode = PermissionMode.DEFAULT
    granted_tools: set[str] = field(default_factory=set)

    def evaluate(self, request: PermissionRequest) -> PermissionDecision:
        """Return the decision for one tool call.

        Rules run in this order; the first one that applies wins:
          0. a hard deny rule matches       -> DENY (nothing can override it)
          1. base decision from risk level  -> ALLOW / ASK / DENY (unknown risk)
          2. read-only mode and not SAFE    -> DENY
          3. granted earlier and ASK        -> ALLOW
          4. auto-approve mode and ASK      -> ALLOW
          5. otherwise                      -> the base decision

        Read-only: never changes the policy's state, so asking twice always
        gives the same answer.
        """
        name = request.tool_name

        hard_deny_reason = self._hard_deny_reason(request)
        if hard_deny_reason is not None:
            return PermissionDecision(
                verdict=Verdict.DENY,
                reason=f"{name} blocked by hard security policy: {hard_deny_reason}",
            )

        decision = self._decide_by_risk(request)

        is_read_only = self.mode is PermissionMode.READ_ONLY
        if is_read_only and request.risk is not RiskLevel.SAFE:
            return PermissionDecision(
                verdict=Verdict.DENY,
                reason=f"{name} blocked: session is in read-only mode",
            )

        if self._is_granted(request) and decision.verdict is Verdict.ASK:
            return PermissionDecision(
                verdict=Verdict.ALLOW,
                reason=f"{name} approved earlier for this session",
            )

        # Auto-approve only answers "yes" to questions; it never overrides a DENY.
        is_auto_approve = self.mode is PermissionMode.AUTO_APPROVE
        if is_auto_approve and decision.verdict is Verdict.ASK:
            return PermissionDecision(
                verdict=Verdict.ALLOW,
                reason=f"{name} auto-approved (auto-approve mode)",
            )

        return decision

    def grant(self, request: PermissionRequest) -> None:
        """Record the user's "always" answer for this tool.

        This is the only method that changes the policy's state.
        """
        self.granted_tools.add(request.tool_name)

    def _is_granted(self, request: PermissionRequest) -> bool:
        """True if the user already said 'always' for this request."""
        return request.tool_name in self.granted_tools

    def _decide_by_risk(self, request: PermissionRequest) -> PermissionDecision:
        """The base decision from risk level alone, before modes apply."""
        name = request.tool_name
        match request.risk:
            case RiskLevel.SAFE:
                return PermissionDecision(
                    verdict=Verdict.ALLOW,
                    reason=f"{name} is read-only",
                )
            case RiskLevel.MUTATING:
                return PermissionDecision(
                    verdict=Verdict.ASK,
                    reason=f"{name} can change files and needs your approval",
                )
            case RiskLevel.DANGEROUS:
                return PermissionDecision(
                    verdict=Verdict.ASK,
                    reason=f"{name} can run any command on your system and needs your approval",
                )
            case _:
                return PermissionDecision(
                    verdict=Verdict.DENY,
                    reason=f"{name} has an unknown risk level {request.risk!r}; denied to be safe",
                )

    def _hard_deny_reason(self, request: PermissionRequest) -> str | None:
        """The reason a hard deny rule blocks this call, or None if none match."""
        if request.tool_name != SHELL_TOOL:
            return None

        command = str(request.arguments.get(SHELL_ARG, ""))
        normalized = _normalize_command(command)

        for pattern, reason in HARD_DENY_RULES:
            if pattern.search(normalized):
                return reason

        return None