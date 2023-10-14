import logging

from pydantic import BaseModel
from instructor.distil import Instructions

logging.basicConfig(level=logging.INFO)

# Usage
instructions = Instructions(
    name="three_digit_multiply",
    finetune_format="messages",
    log_handlers=[
        logging.FileHandler("math_finetunes.jsonl"),
    ],
)


class Response(BaseModel):
    a: int
    b: int
    result: int


@instructions.distil
def fn(a: int, b: int, c: str) -> Response:
    """_summary_

    Args:
        a (int): _description_
        b (int): _description_
        c (str): _description_

    Returns:
        Response: _description_
    """
    resp = a + b
    return Response(a=a, b=b, result=resp)


if __name__ == "__main__":
    import random

    # A log will look like this:
    log_line = {
        "messages": [
            {
                "role": "system",
                "content": 'Predict the results of this function:\n\ndef fn(a: int, b: int, c: str) -> __main__.Response\n"""\n_summary_\n\nArgs:\n    a (int): _description_\n    b (int): _description_\n    c (str): _description_\n\nReturns:\n    Response: _description_\n"""',
            },
            {"role": "user", "content": 'Return fn(133, b=539, c="hello")'},
            {
                "role": "assistant",
                "function_call": {
                    "name": "Response",
                    "arguments": '{"a":133,"b":539,"result":672}',
                },
            },
        ],
        "functions": [
            {
                "name": "Response",
                "description": "Correctly extracted `Response` with all the required parameters with correct types",
                "parameters": {
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "result": {"type": "integer"},
                    },
                    "required": ["a", "b", "result"],
                    "type": "object",
                },
            }
        ],
    }

    for _ in range(10):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        print("returning", fn(a, b=b, c="hello"))
