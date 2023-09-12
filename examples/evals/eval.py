from collections import defaultdict
import json
import logging
from enum import Enum
from pprint import pprint

from pydantic import ValidationError
from summary_stats import StreamingAccumulatorManager
import models as m


class Status(Enum):
    IS_JSON = "_is_json_"
    IS_VALID = "_is_valid_"
    VALIDATION_ERROR = "_validation_error_"


def process_line(eval_manager, line, index):
    try:
        obj = json.loads(line)
        eval_manager.accumulator[Status.IS_JSON.value].update(index, True)

        try:
            obj = m.MultiSearch.model_validate(obj)
            eval_manager.update(index, obj.model_dump())
            eval_manager.accumulator[Status.IS_VALID.value].update(index, True)

        except ValidationError as e:
            eval_manager.accumulator[Status.IS_VALID.value].update(index, False)
            process_validation_error(eval_manager, e, index)

    except json.JSONDecodeError:
        eval_manager.accumulator[Status.IS_JSON.value].update(index, False)


def process_validation_error(eval_manager, error, index):
    for err in error.errors():
        path = (
            "$."
            + ".".join(
                [str(x) if not isinstance(x, int) else "[*]" for x in err["loc"]]
            )
            + "."
            + err["type"]
        )
        eval_manager.accumulator[Status.VALIDATION_ERROR.value].update(index, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    eval_manager = StreamingAccumulatorManager()

    with open("test.jsonl") as f:
        lines = f.readlines()

        for ii, line in enumerate(lines):
            process_line(eval_manager, line, ii)

    pprint(eval_manager.summarize())
