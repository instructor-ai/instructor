import datetime
import json
import os
import uuid
import requests
import logging

from pydantic import BaseModel

from instructor.distil import Instructions


class DatasetHandler(logging.Handler):
    def __init__(self, dataset_name=None):
        super().__init__()
        self.url = os.environ.get("INSTRUCTOR_URL")
        self.api_key = os.environ.get("INSTRUCTOR_KEY")
        self.formatter = logging.Formatter("%(message)s")
        self.dataset_name = dataset_name
        self.uuid = str(uuid.uuid4())

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = json.loads(self.format(record))

        new_entry = {
            "record": log_entry,
            "dataset_name": self.dataset_name,
            "batch_id": self.uuid,
            "created_at": datetime.datetime.now().isoformat(),
        }

        try:
            response = requests.post(self.url, data=json.dumps(new_entry))
            if response.status_code != 200:
                self.handleError(record)
        except Exception:
            self.handleError(record)


logging.basicConfig(level=logging.INFO)

# Usage
instructions = Instructions(
    name="test_distil",
    log_handlers=[
        logging.FileHandler("finetunes.jsonl"),
        DatasetHandler("finetunes_fo_test_distil"),
    ],
)


class Response(BaseModel):
    a: int
    b: int
    result: int


@instructions.distil
def fn(a: int, b: int) -> Response:
    resp = a + b
    return Response(a=a, b=b, result=resp)


if __name__ == "__main__":
    import random

    for _ in range(10):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        print("returning", fn(a, b))
