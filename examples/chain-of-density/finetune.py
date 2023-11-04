from typing import List
from chain_of_density import summarize_article
import csv
import logging
import instructor
from itertools import islice
from pydantic import BaseModel

instructor.patch()

logging.basicConfig(level=logging.INFO)

instructions = instructor.Instructions(
    name="Chain Of Density",
    finetune_format="messages",
    # log handler is used to save the data to a file
    # you can imagine saving it to a database or other storage
    # based on your needs!
    log_handlers=[logging.FileHandler("generated.jsonl")],
)


class GeneratedSummary(BaseModel):
    summary: str


@instructions.distil
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: List[str] = summarize_article(text)
    print(summary_chain)
    return GeneratedSummary(summary=summary_chain[-1])


# Read in the csv file we have
with open("output.csv", "r") as file:
    reader = csv.reader(file)

    for article, summary in islice(reader, 1, 10):
        distil_summarization(article)
