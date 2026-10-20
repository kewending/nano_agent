from datetime import datetime
import ast
import operator
import inspect
import json
import subprocess
import os
from typing import Any, Callable
from nano_agent.config import (
    COMMAND_TIMEOUT_SECONDS,
    MAX_COMMAND_OUTPUT_CHARS,
    DEFAULT_FILE_READ_LIMIT,
    IGNORED_DIR_NAMES
)
TOOLS: list[dict[str, Any]] = []
TOOL_REGISTRY: dict[str, Callable] = {}

PYTHON_TO_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

def tool(func: Callable) -> Callable:
    """Decorator that automatically registers a function and creates its OpenAI schema."""
    schema = function_to_schema(func)
    TOOLS.append(schema)
    TOOL_REGISTRY[func.__name__] = func
    return func

def function_to_schema(func: Callable):
    """Generate Tool Schema JSON from a Python function signature and docstrings."""
    name = func.__name__
    doc = inspect.getdoc(func) or "No description provided."

    sig = inspect.signature(func)
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        # 1. Determine JSON type from Python type annotation
        param_type = PYTHON_TO_JSON_TYPES.get(param.annotation, "string")
        properties[param_name] = {
            "type": param_type,
        }
        # 2. If it has no default value, it is REQUIRED
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return  {
        "type": "function",
        "function": {
            "name": name,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

def execute_tool(name: str, arguments_json: str) -> tuple[str, bool]:
    """Execute a registered tool by name with raw JSON arguments."""
    # Step A: Check if the Tool Exists
    func = TOOL_REGISTRY.get(name)
    if not func:
        return f"Error: Tool '{name}' is not recognized.", True

    # Step B: Parse the JSON Arguments Safely
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return f"Error: Malformed JSON arguments: {arguments_json}", True

    # Step C: Call the Function with Keyword Unpacking
    try:
        result = func(**args)
        return str(result), False
    except Exception as e:
        return f"Error executing '{name}': {e}", True


# ---------------------------------------------------------------------------
# Utility Tools
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """
    Returns the current local time along with the system timezone info.
    """
    local_time = datetime.now().astimezone()
    return local_time.strftime("%Y-%m-%d %H:%M:%S %Z (%z)")

@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression string safely and returns the result as a string.
    Supports +, -, *, /, //, %, **, parentheses, and unary signs.
    """
    # Map AST operator nodes to Python operator functions
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):  # Numbers (int, float)
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):   # Binary operations (e.g., a + b)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](_eval(node.left), _eval(node.right))
            raise TypeError(f"Unsupported binary operator: {op_type}")

        elif isinstance(node, ast.UnaryOp): # Unary operations (e.g., -5)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](_eval(node.operand))
            raise TypeError(f"Unsupported unary operator: {op_type}")

        else:
            raise TypeError("Unsupported expression structure")

    try:
        # Parse expression into an AST tree in 'eval' mode
        tree = ast.parse(expression.strip(), mode='eval')
        result = _eval(tree.body)

        # Clean up floating point display if it's a whole number (e.g., 10.0 -> 10)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))

        return str(result)

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Coding Agent Tools 
# ---------------------------------------------------------------------------

@tool
def read_file(filepath: str, offset: int = 1, limit: int = DEFAULT_FILE_READ_LIMIT) -> str:
    """
    Reads content from a file with line numbers.
    Use offset (1-based index) and limit to paginate through large files.
    """
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' does not exist."
    if os.path.isdir(filepath):
        return f"Error: '{filepath}' is a directory, not a file. Use list_directory instead."

    lines: list[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        for line_num, line in enumerate(file, start=1):
            if line_num < offset:
                continue
            if line_num >= offset + limit:
                break
            lines.append(f"{line_num:4d} | {line}")

    if not lines and offset > 1:
        return f"File '{filepath}' has fewer than {offset} lines."
    return "".join(lines) if lines else "<empty file>"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Creates a new file or overwrites an existing file with the provided content.
    Automatically creates any missing parent directories.
    """
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

    return f"Successfully wrote {len(content)} characters to '{filepath}'."


@tool
def edit_file(filepath: str, target_content: str, replacement_content: str) -> str:
    """
    Replaces a specific block of text ('target_content') with 'replacement_content' in an existing file.
    The target_content must match exactly once in the file to prevent ambiguous edits.
    Include 2-3 lines of surrounding context to ensure uniqueness.
    """
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' does not exist."
    if os.path.isdir(filepath):
        return f"Error: '{filepath}' is a directory, not a file."

    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

    match_count = content.count(target_content)
    if match_count == 0:
        return (
            f"Error: target_content was not found in '{filepath}'. "
            "Ensure indentation, spacing, and exact lines match the file."
        )
    if match_count > 1:
        return (
            f"Error: target_content matched {match_count} times in '{filepath}'. "
            "Please provide more surrounding lines of context to make the target unique."
        )

    updated_content = content.replace(target_content, replacement_content, 1)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(updated_content)

    return f"Successfully updated '{filepath}'."


@tool
def list_directory(path: str = ".") -> str:
    """
    Lists files and subdirectories in a given directory path.
    Automatically filters out noisy build and dependency directories (.git, .venv, __pycache__).
    """
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is a file, not a directory. Use read_file instead."

    entries: list[str] = []

    try:
        with os.scandir(path) as it:
            # Sort directories first, then files alphabetically
            sorted_entries = sorted(it, key=lambda e: (not e.is_dir(), e.name.lower()))
            for entry in sorted_entries:
                if entry.name in IGNORED_DIR_NAMES:
                    continue
                if entry.is_dir():
                    entries.append(f"[DIR]  {entry.name}/")
                else:
                    try:
                        size = entry.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f} MB"
                        entries.append(f"[FILE] {entry.name:<30} ({size_str})")
                    except OSError:
                        entries.append(f"[FILE] {entry.name}")
    except PermissionError:
        return f"Error: Permission denied accessing '{path}'."

    if not entries:
        return f"Directory '{path}' is empty (or only contains ignored directories)."

    return "\n".join(entries)


@tool
def execute_command(command: str) -> str:
    """
    Runs a shell command on the operating system and captures its exit code, stdout, and stderr.
    Useful for running tests, checking git status, inspecting files, and running scripts.
    """

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr

    if not output.strip():
        output = "<No output produced>"

    if len(output) > MAX_COMMAND_OUTPUT_CHARS:
        output = (
            output[:MAX_COMMAND_OUTPUT_CHARS]
            + f"\n... [Output truncated to {MAX_COMMAND_OUTPUT_CHARS} characters] ..."
        )

    return f"[Exit code {proc.returncode}]\n{output}"