---
draft: False
date: 2024-04-02
tags:
  - RAG
  - Embeddings
authors:
  - ivanleomk
  - jxnl
---

## Introduction

Retrieval is the life blood of any RAG application - it determines what context your model sees and ultimately, the quality of the results. Without an easy way to monitor the quality of our retrieval algorithm, we might as well be leaving our application's performance to pure chance.

In this article, we'll walk you through how to use Instructor to generate synthetic data. We'll then chunk and embed some Paul Graham essays using `lancedb`. Next, we'll showcase two useful metrics that we can use to track the performance of our retrieval algorithm before concluding with some interesting improvements to iteratively generate harder evaluation datasets.

As usual, the code that we're using is avaliable inside the `examples/synthethic-evals` folder. We've also included some Paul Graham essays for use in our RAG application.

Let's start by first installing the necessary libraries

```bash
pip install instructor openai scikit-learn rich lancedb tqdm
```

## Generating Evaluation Data

Given a text-chunk, we can use Instructor to generate a corresponding question using the content of the question. This means that when we make a query using that question, our text chunk is ideally going to be the first source returned by our retrieval algorithm.

We can represent this desired result using a simple `pydantic` BaseModel.

### Defining a Data Model

```python
class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        ..., description="The reasoning process leading to the answer.", exclude=True
    )
    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")
```

<!-- more -->

??? info "Excluding Fields"

    When defining a Pydantic Base model, we can specify specific fields to be excluded when we convert our model to a json object. 

    ```python
    >> data = QuestionAnswerPair(chain_of_thought="This is fake", question="Fake question", answer="Fake Answer")
    >> print(data.model_dump_json(indent=2))
        {
            "question": "Fake question",
            "answer": "Fake Answer"
        }
    ```

    This is useful when we have intermediate values/states that we might want to avoid including within the serialized json such as a chain of thought reasoning.

Now that we have a defined data model, we can use `Instructor` to take in a desired text chunk and return a question that is specifically tailored to that text chunk.

```python
from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI
from asyncio import run
from typing import List
from rich import table
from tqdm.asyncio import tqdm_asyncio as asyncio

client = instructor.patch(AsyncOpenAI())


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        ..., description="The reasoning process leading to the answer.", exclude=True
    )
    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")


async def generate_question_answer_pair(chunk: str):
    return await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a world class algorithm that excels at generating great questions that can be only answered by a specific text that will soon be passed to you. ",
            },
            {
                "role": "assistant",
                "content": f"Generate a question and answer pair that uses information and content that is specific to the following text chunk, including a chain of thought:\n\n{chunk}",
            },
        ],
        response_model=QuestionAnswerPair,
    )
```

Note here that we're generating our questions using `async` functions. These allow our calls to OpenAI to be made in parallel so that we don't have to wait for them to be made sequentially and speeds up our execution time significantly. 

<!-- more -->

??? info "Using Async Functions"
    To learn more about using `asyncio` with Instructor, checkout our guide to Batch Processing [here](../posts/learn-async.md) where we show examples demonstrating when to use each method and more.

We can see the results of our function by running it using the code snippet below

```python
if __name__ == "__main__":
    chunk = "The companies at the Google end of the continuum are called startups\
            when they're young. The reason I know about them is that my wife Jessica\
            and I started something called Y Combinator that is basically a startup\
            factory. Since 2005, Y Combinator has funded over 4000 startups.\
            So we know exactly what you need to start a startup, because we\
            've helped people do it for the last 19 years."

    result = run(generate_question_answer_pair(chunk))
    print(result.question)
```

This in turn gives us a sample question of `What is the name of the startup factory founded by the author and his wife, and how many startups has it funded since 2005?`.

### Running our Function

Now that we've defined a data model and function to extract questions from a chunk, we can use a simple wrapper function to generate multiple questions from each of our source chunks in parallel. We can do so using the snippet below.

```python
async def generate_questions(chunks: List[str]):
    coros = [generate_question_answer_pair(chunk) for chunk in chunks]
    return await asyncio.gather(*coros)
```

In this case, we've chosen to use `tqdm`'s asyncio module instead of the native `asyncio` module because of two main reasons 

1. `tqdm` offers an easy way to monitor the result of our question generation with it's native progress bar
2. By importing it in as `asyncio`, we can easily swap it out down the line for the original asyncio library if our needs change

If you want to extend this to a larger batch, consider using the `tenacity` library. With a simple `@retry` decorator, it provides useful features such as exponential backoff, error handling and maximum retries. 

Let's see how our function scales out to more chunks. We'll be using the `rich` library to format the generated output.

```python
if __name__ == "__main__":
    chunks = [
        "The companies at the Google end of the continuum are called startups when they're young. The reason I know about them is that my wife Jessica and I started something called Y Combinator that is basically a startup factory. Since 2005, Y Combinator has funded over 4000 startups. So we know exactly what you need to start a startup, because we've helped people do it for the last 19 years.",
        "All you can know when you start working on a startup is that it seems worth pursuing. You can't know whether it will turn into a company worth billions or one that goes out of business. So when I say I'm going to tell you how to start Google, I mean I'm going to tell you how to get to the point where you can start a company that has as much chance of being Google as Google had of being Google.",
        "Those of you who are taking computer science classes in school may at this point be thinking, ok, we've got this sorted. We're already being taught all about programming. But sorry, this is not enough. You have to be working on your own projects, not just learning stuff in classes. You can do well in computer science classes without ever really learning to program. In fact you can graduate with a degree in computer science from a top university and still not be any good at programming. That's why tech companies all make you take a coding test before they'll hire you, regardless of where you went to university or how well you did there. They know grades and exam results prove nothing.",
    ]

    questions: List[QuestionAnswerPair] = run(generate_questions(chunks))

    table = Table(title="Questions and Sources", show_lines=True)
    table.add_column(
        "Question", style="magenta", justify="left", no_wrap=False, max_width=100
    )
    table.add_column(
        "Original Source", style="cyan", justify="left", no_wrap=False, max_width=100
    )

    for question, chunk in zip(questions, chunks):
        table.add_row(question.question, chunk)

    console = Console()
    console.print(table, justify="center")
```

This in turn gives us a nicely formated table as seen below

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Question                       ┃ Original Source                                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ What is the name of the        │ The companies at the Google end of the continuum   |
│ startup factory mentioned in   │ are called startups when they're young. The reason |
│ the text that has funded over  │ I know about them is that my wife Jessica and I    |
│ 4000 startups since 2005?      │ started something called Y Combinator that is      |
|                                │ basically a startup factory. Since 2005, Y         |
|                                │ Combinator has funded over 4000 startups. So we    |
|                                │ know exactly what you need to start a startup,     |
|                                │ because we've helped people do it for the last 19  |
|                                │ years.                                             |
├────────────────────────────────┼────────────────────────────────────────────────────┤                  
│ What does the text suggest     │ All you can know when you start working on a       |
│ about starting a startup and   │ startup is that it seems worth pursuing. You can't |
│ the uncertainty of its         │ know whether it will turn into a company worth     |
│ success?                       │ billions or one that goes out of business. So when |
|                                │ I say I'm going to tell you how to start Google, I |
|                                │ mean I'm going to tell you how to get to the point |
|                                │ where you can start a company that has as much     |
|                                │ chance of being Google as Google had of being      |
|                                │ Google.                                            |
├────────────────────────────────┼────────────────────────────────────────────────────┤                  
│ Why do tech companies make     │ Those of you who are taking computer science       |
│ candidates take a coding test  │ classes in school may at this point be thinking,   |
│ before hiring them?            │ ok, we've got this sorted. We're already being     |
|                                │ taught all about programming. But sorry, this is   |
|                                │ not enough. You have to be working on your own     |
|                                │ projects, not just learning stuff in classes. You  |
|                                │ can do well in computer science classes without    |
|                                │ ever really learning to program. In fact you can   |
|                                │ graduate with a degree in computer science from a  |
|                                │ top university and still not be any good at        |
|                                │ programming. That's why tech companies all make    |
|                                │ you take a coding test before they'll hire you,    |
|                                │ regardless of where you went to university or how  |
|                                │ well you did there. They know grades and exam      |
|                                │ results prove nothing.                             |
└────────────────────────────────┴────────────────────────────────────────────────────┘
```

We can see that for each individual chunk of text that we have, we now have a well formatted question that is directly targetted at the content of the text itself. 

## Scaling Our Chunking

Now that we've written a simple function to generate questions from chunks, let's generate some text chunks from some Paul Graham essays. We've included 5 essays inside the `data` folder alongside our code at `/examples/synthethic-evals`. A good fit for this is the `lancedb` library which natively supports Pydantic.

This gives us a Vector Database which we can define entirely using `Pydantic` base models that also handles the batching for us out of the box nicely.

### Data Models

We need a simple data model to represent a chunk of text - this is simply an excerpt from one of the essays that we've provided.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)

class TextChunk(LanceModel):
    chunk_id: str
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)
```

Note here that a `LanceModel` is a `LanceDB` specific class which is based off a simple Pydantic `BaseModel`. It just adds some `LanceDB` specific functionality so that it works with the library. We've also defined a new field called `vector` which will automatically create an embedding vector using the OpenAI `text-embedding-3-large` model for us when we do the insertions.

Creating a new `LanceDB` vector database is simple - all we do is to use the `connect` method provided by the library and `lancedb` handles the rest.

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb import connect

openai = get_registry().get("openai").create(name="text-embedding-3-large", dim=256)


class TextChunk(LanceModel):
    chunk_id: str
    text: str = openai.SourceField()
    vector: Vector(openai.ndims()) = openai.VectorField(default=None)


if __name__ == "__main__":
    db_path = "./db"
    table_name = "pg"
    db = connect(db_path)
    db_table = db.open_table(table_name)
```

This creates a new folder for us at the path `./db` which will store all of our folder metadata and data.

### Chunking our Data

Now that we have created a new `lancedb` database locally, we need to do two things

1. Read in the text data from the individual `.md` files containing the essays
2. Split the text data by `\n`, generate an embedding for each and a corresponding hash using the `hashlib` library
3. Split the chunks into batches and insert it into our lancedb in batches

We can perform the first part by writing two simple iterators - one that returns a list of all the `.md` files in a directory and another that generates chunks from these files. This is relatively simple in python.

```python
from typing import Iterable, List
from pathlib import Path
import hashlib

def read_files(path: str) -> Iterable[str]:
    path_obj = Path(path)
    for file in path_obj.rglob(f"*.md"):
        yield file


def generate_chunks(docs: Iterable[List[Path]]):
    for doc in docs:
        with open(doc, "r", encoding="utf-8") as file:
            content = file.read()
            for chunk in content.split("\n"):
                if not chunk:
                    continue
                yield TextChunk(
                    text=chunk, chunk_id=hashlib.md5(chunk.encode("utf-8")).hexdigest()
                )
```

We can then batch each of these individual group of text chunks by using another iterator as 

```python
def batch_items(chunks: List[TextChunk], batch_size: int = 20):
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
```

Finally, we can combine all of these together into a simple function as seen below.

```python
if __name__ == "__main__":
    db_path = "./db"
    table_name = "pg"
    data_path = "./data"

    db = connect(db_path)
    db_table = db.create_table(table_name, exist_ok=True, schema=TextChunk)

    files = read_files(data_path)
    chunks = generate_chunks(files)

    batched_chunks = batch_items(chunks, batch_size=20)

    for batch in batched_chunks:
        db_table.add(batch)
```

What's really neat about this entire setup is that `lancedb` is handling the automatic batching of embedding calls and iterators for us. We don't need to manually generate the embeddings and then create new objects with these embeddings.

### Retrieving Data

TODO: Add in a section for how to embed a query, then scale up to embed a bunch of queries before seeing what data is retrieved

## Evaluations

There are two useful evaluations which we can use when evaluating the quality of retrievals - NDCG and MRR. Intuitively, since we used a single chunk to generate a question, we want to measure how often it appears as the top result. 

Eventually, as we utilise more complex methods to generate our questions - either by combining multiple chunks or writing more complex queries, our approach will change. But for now, these two metrics are great for quickly determining how good our retrieval is performing.

??? info "Evaluations"
    
    For a more complex and in-depth look into different metrics, consider looking at [this article written by Jason on evaluating Machine Learning Systems](https://jxnl.github.io/blog/writing/2024/02/05/when-to-lgtm-at-k/#mean-reciprocal-rank-mrr-k)

### Chunking 

TODO: Quick guide to what we need to do to evaluate NDCG and MRR rankings then run a larger job


### Results

TODO: Look at the nicely formatted results and maybe identify some general trends

## Conclusion

TODO: Look at some improvements (Eg. randomly select two chunks, find a chunk that summarizes or look at a new way of question answering ( What is the difference in opinion over the last 3 years for topic X))

TODO: Write nice conclusion