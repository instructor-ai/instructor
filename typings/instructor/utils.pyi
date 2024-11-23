"""Type stubs for instructor.utils."""
from typing import Generator, AsyncGenerator, Any, Iterable

def extract_json_from_stream(
    chunks: Iterable[str],
) -> Generator[str, None, None]: ...

async def extract_json_from_stream_async(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]: ...
