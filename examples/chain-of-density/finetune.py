from typing import List
from chain_of_density import summarize_article, compute_metrics
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


# @instructions.distil
# @instructions.distil(model="ft:gpt-3.5-turbo-0613:personal::8HQgyZBo", mode="dispatch")
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: List[str] = summarize_article(text)
    return GeneratedSummary(summary=summary_chain[-1])


# Read in the csv file we have
with open("output.csv", "r") as file:
    reader = csv.reader(file)

    summaries = []
    for article, summary in islice(reader, 30, 35):
        summaries.append(distil_summarization(article))

    ttl_tokens = 0
    ttl_entities = 0

    for summary in summaries:
        num_tokens, num_entities, ET_ratio = compute_metrics(summary.summary)
        ttl_tokens += num_tokens
        ttl_entities += num_entities
        print(
            f"Token Count: {num_tokens}, Entity Count: {num_entities}, E/T : {ET_ratio}"
        )

    print(f"FINAL ET: {ttl_entities/ttl_tokens}")
