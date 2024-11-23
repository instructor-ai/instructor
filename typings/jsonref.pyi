"""Type stubs for jsonref."""
from typing import Any, Optional, Union, Dict, List, Mapping

class JsonRef:
    """Type stub for JsonRef class."""
    pass

def loads(
    s: str,
    base_uri: str = "",
    loader: Optional[Any] = None,
    jsonschema: bool = False,
    load_on_repr: bool = True,
    merge_props: bool = False,
    proxies: bool = True,
    lazy_load: bool = True,
    **kwargs: Any
) -> Union[JsonRef, List[Any], str, Dict[str, Any], Mapping[str, Any]]: ...
