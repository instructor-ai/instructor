"""Type stubs for jsonref."""
from typing import Any
from collections.abc import Mapping

class JsonRef:
    """Type stub for JsonRef class."""
    pass

def loads(
    s: str,
    base_uri: str = "",
    loader: None | Any = None,
    jsonschema: bool = False,
    load_on_repr: bool = True,
    merge_props: bool = False,
    proxies: bool = True,
    lazy_load: bool = True,
    **kwargs: Any
) -> JsonRef | list[Any] | str | dict[str, Any] | Mapping[str, Any]: ...
