"""Type stubs for jiter."""
from typing import Any, Optional, Union, Dict

def from_json(
    data: Union[str, bytes],
    *,
    partial_mode: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]: ...
