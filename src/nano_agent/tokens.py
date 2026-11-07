import json
from typing import Any
import tiktoken

# Token heuristic count
_ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))

def count_payload_tokens(payload: Any) -> int:
    """Serialize any dict/object to JSON and count its tokens."""
    if isinstance(payload, str):
        return count_tokens(payload)
    return count_tokens(json.dumps(payload, ensure_ascii=False))