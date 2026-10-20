from datetime import datetime
import ast
import operator
import inspect
import json
from typing import Any, Callable

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
    """Generate Tool Schema Json from function"""
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


@tool
def get_current_time() -> str:
    """
    Returns the current local time along with the system timezone info.
    """
    # Get current time adjusted to the system's local timezone
    local_time = datetime.now().astimezone()
    
    # Format cleanly: YYYY-MM-DD HH:MM:SS TimezoneName (Offset)
    # Example: 2026-10-18 08:54:00 AEDT (+1100)
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
            raise TypeError(f"Unsupported expression structure")

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