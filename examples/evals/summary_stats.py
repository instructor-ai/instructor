# Modified StreamingAccumulator class with self.value and self.str_length as lists

from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, Union, List
import numpy as np
from pydantic import BaseModel, Field


class StreamingAccumulator:
    counter: Counter = Field(default_factory=Counter)
    min: float = float("inf")
    max: float = float("-inf")
    sum: float = 0
    squared_sum: float = 0
    unique_values: set = Field(default_factory=set)
    missing_values: int = 0
    str_min_length: float = float("inf")
    str_max_length: float = float("-inf")
    str_sum_length: float = 0
    str_squared_sum_length: float = 0
    value: List[Any] = Field(default_factory=list)  # Added back as a list
    str_length: List[int] = Field(default_factory=list)  # Added back as a list
    reverse_lookup: defaultdict = defaultdict(list)

    def __init__(self):
        self.counter = Counter()
        self.min = float("inf")
        self.max = float("-inf")
        self.sum = 0
        self.squared_sum = 0
        self.unique_values = set()
        self.missing_values = 0
        self.str_min_length = float("inf")
        self.str_max_length = float("-inf")
        self.str_sum_length = 0
        self.str_squared_sum_length = 0
        self.value = []
        self.str_length = []
        self.reverse_lookup = defaultdict(list)

    def update(self, index: Any, value: Any) -> None:
        """Update statistics with a new value."""

        if isinstance(value, (int, str, bool)):
            self.counter[value] += 1
            self.unique_values.add(value)
            self.value.append(value)
            self.reverse_lookup[value].append(index)

        if value is None or value == "":
            self.missing_values += 1
            return

        if isinstance(value, (int, float)):
            self.min = min(self.min, value)
            self.max = max(self.max, value)
            self.sum += value
            self.squared_sum += value**2

        if isinstance(value, str):
            str_len = len(value)
            self.str_length.append(str_len)  # Append the string length to the list
            self.str_min_length = min(self.str_min_length, str_len)
            self.str_max_length = max(self.str_max_length, str_len)
            self.str_sum_length += str_len
            self.str_squared_sum_length += str_len**2

    def summarize(self, key_name=None) -> Dict[str, Union[int, float, dict]]:
        if key_name is None:
            key_name = ""

        n = sum(self.counter.values())
        summaries = {}
        summaries["counter"] = self.counter
        summaries["unique_count"] = len(self.unique_values)
        summaries["missing_values"] = self.missing_values
        summaries["_reverse_lookup"] = dict(self.reverse_lookup)

        if n > 0:
            if all(isinstance(value, (bool)) for value in self.unique_values):
                summaries["mean"] = self.sum / n
                return summaries

            if all(isinstance(value, (int, float)) for value in self.unique_values):
                summaries["min"] = self.min
                summaries["max"] = self.max
                summaries["mean"] = self.sum / n
                summaries["std"] = np.sqrt(self.squared_sum / n - (self.sum / n) ** 2)
                return summaries

            if all(
                isinstance(value, str) for value in self.unique_values
            ) and not key_name.startswith("_"):
                summaries["str_min_length"] = self.str_min_length
                summaries["str_max_length"] = self.str_max_length
                summaries["str_mean_length"] = self.str_sum_length / n
                summaries["str_std_length"] = np.sqrt(
                    self.str_squared_sum_length / n - (self.str_sum_length / n) ** 2
                )
                return summaries

        return summaries


class StreamingAccumulatorManager:
    def __init__(self):
        self.accumulator = defaultdict(StreamingAccumulator)

    def update(self, index, data: Any, path: str = "$") -> None:
        """Accumulate values from a nested object."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}"
                self.update(index, value, new_path)
        elif isinstance(data, list):
            new_path = f"{path}[*]"
            for value in data:
                self.update(index, value, new_path)
            length_path = f"{path}.length"
            self.accumulator[length_path].update(index, len(data))
        elif isinstance(data, Enum):
            enum_path = f"{path}.enum"
            self.accumulator[enum_path].update(index, data.value)
        elif path != "$":
            pass
        else:
            self.accumulator[path].update(index, data)

    def summarize(self) -> Dict[str, Dict]:
        """Generate summary statistics for all paths."""
        return {k: v.summarize(key_name=k) for k, v in self.accumulator.items()}
