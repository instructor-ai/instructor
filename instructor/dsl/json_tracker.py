"""
JSON Completeness Tracker for Partial Streaming.

Tracks which parts of accumulated JSON are "closed" (complete) vs "open" (incomplete).
A closed object/array has matching braces/brackets; an open one is still being streamed.

This enables validation to run only on complete sub-objects, avoiding validation
errors on incomplete data during streaming.
"""

from __future__ import annotations


class JsonCompleteness:
    """
    Track completeness of JSON structures during streaming.

    A JSON structure is "complete" if:
    - Objects: start with { and end with }
    - Arrays: start with [ and end with ]
    - Scalars (strings, numbers, booleans, null): always complete once parsed

    Example:
        tracker = JsonCompleteness()

        # Incomplete - missing closing brace
        tracker.analyze('{"name": "Alice", "address": {"city": "NY')
        tracker.is_path_complete("")  # False - root object incomplete
        tracker.is_path_complete("name")  # True - string is complete
        tracker.is_path_complete("address")  # False - nested object incomplete

        # Complete
        tracker.analyze('{"name": "Alice"}')
        tracker.is_path_complete("")  # True - root object complete
    """

    def __init__(self) -> None:
        self._json_str: str = ""
        self._complete_paths: set[str] = set()
        self._path_positions: dict[str, tuple[int, int]] = {}  # path -> (start, end)

    def analyze(self, json_str: str) -> None:
        """
        Analyze a JSON string and determine completeness of each sub-structure.

        Args:
            json_str: The accumulated JSON string (may be incomplete)
        """
        self._json_str = json_str
        self._complete_paths = set()
        self._path_positions = {}

        if not json_str.strip():
            return

        # Parse and track completeness
        self._analyze_structure(json_str, "", 0)

    def _analyze_structure(self, json_str: str, path: str, start_pos: int) -> int:
        """
        Recursively analyze JSON structure and track completeness.

        Returns the position after the current structure, or -1 if incomplete.
        """
        s = json_str[start_pos:].lstrip()
        if not s:
            return -1

        pos = start_pos + (
            len(json_str) - start_pos - len(json_str[start_pos:].lstrip())
        )

        if s[0] == "{":
            return self._analyze_object(json_str, path, pos)
        elif s[0] == "[":
            return self._analyze_array(json_str, path, pos)
        elif s[0] == '"':
            return self._analyze_string(json_str, path, pos)
        elif s[0] in "-0123456789":
            return self._analyze_number(json_str, path, pos)
        elif s.startswith("true"):
            self._mark_complete(path, pos, pos + 4)
            return pos + 4
        elif s.startswith("false"):
            self._mark_complete(path, pos, pos + 5)
            return pos + 5
        elif s.startswith("null"):
            self._mark_complete(path, pos, pos + 4)
            return pos + 4
        else:
            return -1  # Invalid or incomplete

    def _analyze_object(self, json_str: str, path: str, start_pos: int) -> int:
        """Analyze a JSON object. Returns end position or -1 if incomplete."""
        pos = start_pos + 1  # Skip opening {
        first = True

        while pos < len(json_str):
            # Skip whitespace
            while pos < len(json_str) and json_str[pos] in " \t\n\r":
                pos += 1

            if pos >= len(json_str):
                return -1  # Incomplete

            if json_str[pos] == "}":
                # Object is complete
                self._mark_complete(path, start_pos, pos + 1)
                return pos + 1

            if not first:
                if json_str[pos] != ",":
                    return -1  # Invalid
                pos += 1
                # Skip whitespace after comma
                while pos < len(json_str) and json_str[pos] in " \t\n\r":
                    pos += 1
                if pos >= len(json_str):
                    return -1

            first = False

            # Parse key
            if pos >= len(json_str) or json_str[pos] != '"':
                return -1  # Invalid or incomplete

            key_start = pos
            pos = self._skip_string(json_str, pos)
            if pos == -1:
                return -1  # Incomplete string

            key = json_str[key_start + 1 : pos - 1]  # Extract key without quotes

            # Skip whitespace and colon
            while pos < len(json_str) and json_str[pos] in " \t\n\r":
                pos += 1
            if pos >= len(json_str) or json_str[pos] != ":":
                return -1
            pos += 1

            # Parse value
            child_path = f"{path}.{key}" if path else key
            pos = self._analyze_structure(json_str, child_path, pos)
            if pos == -1:
                return -1  # Incomplete value

        return -1  # Incomplete (no closing brace)

    def _analyze_array(self, json_str: str, path: str, start_pos: int) -> int:
        """Analyze a JSON array. Returns end position or -1 if incomplete."""
        pos = start_pos + 1  # Skip opening [
        index = 0
        first = True

        while pos < len(json_str):
            # Skip whitespace
            while pos < len(json_str) and json_str[pos] in " \t\n\r":
                pos += 1

            if pos >= len(json_str):
                return -1  # Incomplete

            if json_str[pos] == "]":
                # Array is complete
                self._mark_complete(path, start_pos, pos + 1)
                return pos + 1

            if not first:
                if json_str[pos] != ",":
                    return -1  # Invalid
                pos += 1
                # Skip whitespace after comma
                while pos < len(json_str) and json_str[pos] in " \t\n\r":
                    pos += 1
                if pos >= len(json_str):
                    return -1

            first = False

            # Parse element
            child_path = f"{path}[{index}]"
            pos = self._analyze_structure(json_str, child_path, pos)
            if pos == -1:
                return -1  # Incomplete element

            index += 1

        return -1  # Incomplete (no closing bracket)

    def _analyze_string(self, json_str: str, path: str, start_pos: int) -> int:
        """Analyze a JSON string. Returns end position or -1 if incomplete."""
        pos = self._skip_string(json_str, start_pos)
        if pos != -1:
            self._mark_complete(path, start_pos, pos)
        return pos

    def _skip_string(self, json_str: str, start_pos: int) -> int:
        """Skip a JSON string, handling escapes. Returns position after closing quote or -1."""
        pos = start_pos + 1  # Skip opening quote
        while pos < len(json_str):
            c = json_str[pos]
            if c == "\\":
                pos += 2  # Skip escape sequence
            elif c == '"':
                return pos + 1  # Found closing quote
            else:
                pos += 1
        return -1  # Incomplete string

    def _analyze_number(self, json_str: str, path: str, start_pos: int) -> int:
        """Analyze a JSON number. Returns end position or -1 if incomplete."""
        pos = start_pos

        # Optional minus
        if pos < len(json_str) and json_str[pos] == "-":
            pos += 1

        # Integer part
        if pos >= len(json_str):
            return -1
        if json_str[pos] == "0":
            pos += 1
        elif json_str[pos] in "123456789":
            pos += 1
            while pos < len(json_str) and json_str[pos] in "0123456789":
                pos += 1
        else:
            return -1

        # Fractional part
        if pos < len(json_str) and json_str[pos] == ".":
            pos += 1
            if pos >= len(json_str) or json_str[pos] not in "0123456789":
                return -1  # Incomplete fraction
            while pos < len(json_str) and json_str[pos] in "0123456789":
                pos += 1

        # Exponent part
        if pos < len(json_str) and json_str[pos] in "eE":
            pos += 1
            if pos < len(json_str) and json_str[pos] in "+-":
                pos += 1
            if pos >= len(json_str) or json_str[pos] not in "0123456789":
                return -1  # Incomplete exponent
            while pos < len(json_str) and json_str[pos] in "0123456789":
                pos += 1

        # Check if we're at a valid terminator (or end of partial JSON)
        if pos < len(json_str) and json_str[pos] not in " \t\n\r,}]":
            return -1  # Number continues or is invalid

        self._mark_complete(path, start_pos, pos)
        return pos

    def _mark_complete(self, path: str, start_pos: int, end_pos: int) -> None:
        """Mark a path as complete."""
        self._complete_paths.add(path)
        self._path_positions[path] = (start_pos, end_pos)

    def is_path_complete(self, path: str) -> bool:
        """
        Check if the sub-structure at the given path is complete.

        Args:
            path: Dot-separated path (e.g., "user.address.city", "items[0]")
                  Use "" for root object.

        Returns:
            True if the structure at path is complete (closed), False otherwise.
        """
        return path in self._complete_paths

    def get_complete_paths(self) -> set[str]:
        """Return all paths that are complete."""
        return self._complete_paths.copy()

    def is_root_complete(self) -> bool:
        """Check if the root JSON structure is complete."""
        return "" in self._complete_paths


def is_json_complete(json_str: str) -> bool:
    """
    Quick check if a JSON string represents a complete structure.

    Uses jiter in strict mode - parsing fails if JSON is incomplete.

    Args:
        json_str: The JSON string to check

    Returns:
        True if the JSON is complete (all braces/brackets matched)
    """
    from jiter import from_json

    if not json_str or not json_str.strip():
        return False
    try:
        from_json(json_str.encode())  # No partial_mode = strict parsing
        return True
    except Exception:
        return False
