from pydantic import BaseModel
from chain_of_density import Summary, summarize_article
import csv
from itertools import islice
import logging
import instructor
import openai

instructor.patch()

openai.api_key = "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"
logging.basicConfig(level=logging.INFO)

instructions = instructor.Instructions(
    name="Chain Of Density",
    finetune_format="messages",
    # log handler is used to save the data to a file
    # you can imagine saving it to a database or other storage
    # based on your needs!
    log_handlers=[logging.FileHandler("summarization.jsonl")],
)


class Multiply(BaseModel):
    a: int
    b: int
    result: int


@instructions.distil
def distil_summarization(text: str) -> Summary:
    summary_chain = summarize_article(text, stream=False)
    generated_summaries = [i for i in summary_chain][0][1]

    # We implement a rudimentary retry with MultiTask for now - if we do not have at least 5 items in the chain, we'll retry the entire
    # chain
    assert len(generated_summaries) >= 5
    return generated_summaries[-1]


# Read in the csv file we have
with open("output.csv", "r") as file:
    reader = csv.reader(file)

    # Skip the header row
    next(reader)
    for article, summary in islice(reader, 18):
        for _ in range(3):
            try:
                generated_summary = distil_summarization(article)
                break
            except Exception as e:
                print(f"Failed to generate summary due to {e}")
