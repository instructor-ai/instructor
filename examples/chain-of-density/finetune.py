from typing import List
from chain_of_density import summarize_article
import csv
import logging
import instructor
from pydantic import BaseModel
from openai import OpenAI

client = instructor.patch(OpenAI())

logging.basicConfig(level=logging.INFO)

instructions = instructor.Instructions(
    name="Chain Of Density",
    finetune_format="messages",
    log_handlers=[logging.FileHandler("summarization22.jsonl")],
)


class GeneratedSummary(BaseModel):
    summary: str


@instructions.distil
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: List[str] = summarize_article(text)
    return GeneratedSummary(summary=summary_chain[-1])


# Read in the csv file we have
with open("test.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for article, summary in reader:
        distil_summarization(article)
