import json
from instructor.utils import extract_json_from_codeblock, extract_json_from_stream


def test_extract_json_from_codeblock():
    example = """
    Here is a response

    ```json
    {
        "key": "value"
    }    
    ```
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_extract_json_from_codeblock_no_end():
    example = """
    Here is a response

    ```json
    {
        "key": "value"
    }    
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_extract_json_from_codeblock_no_start():
    example = """
    Here is a response

    {
        "key": "value"
    }    
    """
    result = extract_json_from_codeblock(example)
    assert json.loads(result) == {"key": "value"}


def test_stream_json():
    text = """here is the json for you! 
    
    ```json
    {
        "key": "value",
        "another_key": {"key": "value"}
    }
    ```

    What do you think?
    """

    def batch_strings(chunks, n=2):
        batch = ""
        for chunk in chunks:
            for char in chunk:
                batch += char
                if len(batch) == n:
                    yield batch
                    batch = ""
        if batch:  # Yield any remaining characters in the last batch
            yield batch

    result = json.loads(
        "".join(list(extract_json_from_stream(batch_strings(text, n=3))))
    )
    assert result == {"key": "value", "another_key": {"key": "value"}}
