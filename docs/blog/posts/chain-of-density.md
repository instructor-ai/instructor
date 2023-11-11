---
draft: False
date: 2023-11-05
tags:
  - pydantic
  - validation
  - chain of density
  - finetuneing
  - gpt-3.5-turbo
  - distilation
authors:
  - ivanleomk
  - jxnl
---

# Better Summaries by Finetuning Chain of Density

> Discover how to distil an interative method like chain of density into a single finetune.

In this article, we'll guide you through implementing the original Chain of Density method using Instructor, then show how to distile a GPT 3.5 model to match GPT-4's iterative summarization capabilities. Using these methods were able to increase latency by 40x, reduce costs by 10x and maintain entity density. Showing massive efficiency gains by finetuning and distiling capabilities into specialized models.

By the end you'll end up with a GPT 3.5 model, (fine-tuned using Instructor's great tooling), capable of producing summaries that rival the effectiveness of Chain of Density. As always, all code is readily available in our `examples/chain-of-density` folder in our repo for your reference.

??? abstract "Datasets and Colab Notebook"

    We've also uploaded all our generated data to Hugging Face [here](https://huggingface.co/datasets/ivanleomk/gpt4-chain-of-density) for you to use if you'd like to try reproducing these experiments. We've also added a [Colab Instance](https://colab.research.google.com/drive/1iBkrEh2G5U8yh8RmI8EkWxjLq6zIIuVm?usp=sharing) for you to check our generated values.

## Part 1) Chain of Density

Summarizing extensive texts with AI can be challenging, often relying on inconsistent techniques. Salesforce AI Research's novel method, chain of density, enhances AI-based text summarization, outperforming human-generated summaries.

Initially, an AI produces a summary, then refines it through multiple iterations, adding missing article entities. Each iteration adds new article entities to the summary, keeping length consistent, leading to an entity-dense, informative summary called Chain Of Density.

First introduced by Salesforce's AI Research wing in their paper - [From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting](https://arxiv.org/abs/2309.04269). The team has found that this method is able to consistently beats similar summaries written by human annotators.

??? info "Implementation Details"

    Note that our implementation uses a validator to ensure that the rewritten summary has a minimum length rather than a prompt. As a result, we match the original paper on entity count but not entity density.

### Original Prompt

We can implement the original prompt using `pip install instructor` by breaking down the entire process into smaller api calls. This allows us to introduce validation at each step to ensure that we're getting the results that we want.

??? note "Original Chain of Density Prompt"

    ```
    Article: {{ARTICLE}}

    You will generate increasingly concise, entity-dense summaries of the
    above Article.

    Repeat the following 2 steps 5 times.

    Step 1. Identify 1-3 informative Entities (";" delimited) from the
    Article which are missing from the previously generated summary.
    Step 2. Write a new, denser summary of identical length which covers
    every entity and detail from the previous summary plus the Missing
    Entities.

    A Missing Entity is:
    - Relevant: to the main story.
    - Specific: descriptive yet concise (5 words or fewer).
    - Novel; not in the previous summary.
    - Faithful: present in the Article.
    - Anywhere: located anywhere in the Article.

    Guidelines:
    - The first summary should be long (4-5 sentences, -80 words) yet
    highly non-specific, containing little information beyond the
    entities marked as missing. Use overly verbose language and fillers
    (e.g., "this article discusses") to reach -80 words.
    - Make every word count: re-write the previous summary to improve
    flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative
    phrases like "the article discusses"
    - The summaries should become highly dense and concise yet
    self-contained, e.g., easily understood without the Article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be
    made, add fewer new entities.

    Remember, use the exact same number of words for each summary.

    Answer in JSON. The JSON should be a list (length 5) of dictionaries
    whose keys are "Missing_Entities" and "Denser_Summary"
    ```

<figure markdown>
  ![RAG](img/chain-of-density.png)
  <figcaption>Improved process with Instructor</figcaption>
</figure>

### Data Modelling

#### Initial Summary

Let's start by walking through some of the data models that we'll be using as the `response_model` for our open ai function calls

Firstly, we'll need a data model for the initial summary that we will be generating. We'll take the description of this class straight from the original prompt. Its important to note that these docstrings serve a purpose, they are directly used by the LLM when generating the outputs.

```py
class InitialSummary(BaseModel):
    """
    This is an initial summary which should be long ( 4-5 sentences, ~80 words)
    yet highly non-specific, containing little information beyond the entities marked as missing.
    Use overly verbose languages and fillers (Eg. This article discusses) to reach ~80 words.
    """

    summary: str = Field(
        ...,
        description="This is a summary of the article provided which is overly verbose and uses fillers. It should be roughly 80 words in length",
    )
```

#### Rewritten Summary

We'll also need one additional class to help model the rewritten schema

```py
class RewrittenSummary(BaseModel):
    """
    This is a new, denser summary of identical length which covers every entity
    and detail from the previous summary plus the Missing Entities.

    Guidelines
    - Make every word count : Rewrite the previous summary to improve flow and make space for additional entities
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
    - Missing entities can appear anywhere in the new summary

    An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
    """

    summary: str = Field(
        ...,
        description="This is a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities. It should have the same length ( ~ 80 words ) as the previous summary and should be easily understood without the Article",
    )
    absent: List[str] = Field(
        ...,
        default_factory=list,
        description="this is a list of Entities found absent from the new summary that were present in the previous summary",
    )
    missing: List[str] = Field(
        default_factory=list,
        description="This is a list of 1-3 informative Entities from the Article that are missing from the new summary which should be included in the next generated summary.",
    )
```

!!! tip "Using Pydantic Validators with Instructor"

    For a more in-depth walkthrough on how to use `Pydantic` validators with the `Instructor`
    library, we recommend checking out our previous article on LLM
    validation - [Good LLM Validation is just Good Validation](/instructor/blog/2023/10/23/good-llm-validation-is-just-good-validation/)

Ideally, we'd like for `Missing` to have a length between 1 and 3, `Absent` to be an empty list and for our summary to be around 80 tokens. With `Instructor`, we can implement this logic using native `Pydantic` validators that are simply declared as part of the class itself.

```py hl_lines="3"
@field_validator("summary")
def min_length(cls, v: str):
    tokens = nltk.word_tokenize(v) #(1)!
    num_tokens = len(tokens)
    if num_tokens < 75:
        raise ValueError(
            "The current summary is too short. Please make sure that you generate a new summary that is around 80 words long."
        )
    return v

@field_validator("missing")
def has_missing_entities(cls, missing_entities: List[str]):
    if len(missing_entities) == 0:
        raise ValueError(
            "You must identify 1-3 informative Entities from the Article which are missing from the previously generated summary to be used in a new summary"
        )
    return missing_entities

@field_validator("absent")
def has_no_absent_entities(cls, absent_entities: List[str]):
    absent_entity_string = ",".join(absent_entities)
    if len(absent_entities) > 0:
        print(f"Detected absent entities of {absent_entity_string}")
        raise ValueError(
            f"Do not omit the following Entities {absent_entity_string} from the new summary"
        )
    return absent_entities
```

1.  Similar to the original paper, we utilize the `NLTK` word tokenizer to count the number of tokens within our generated sentences.
    We aim for at least 75 tokens in our generated summary so that we don't lose information.

### Putting it all Together

Now that we have our models and the rough flow figured out, let's implement a function to summarize a piece of text using `Chain Of Density` summarization.

```py hl_lines="4 9-24 38-68"
from openai import OpenAI
import instructor

client = instructor.patch(OpenAI()) #(1)!

def summarize_article(article: str, summary_steps: int = 3):
    summary_chain = []
    # We first generate an initial summary
    summary: InitialSummary = client.chat.completions.create(  # (2)!
        model="gpt-4-0613",
        response_model=InitialSummary,
        messages=[
            {
                "role": "system",
                "content": "Write a summary about the article that is long (4-5 sentences) yet highly non-specific. Use overly, verbose language and fillers(eg.,'this article discusses') to reach ~80 words",
            },
            {"role": "user", "content": f"Here is the Article: {article}"},
            {
                "role": "user",
                "content": "The generated summary should be about 80 words.",
            },
        ],
        max_retries=2,
    )
    prev_summary = None
    summary_chain.append(summary.summary)
    for i in range(summary_steps):
        missing_entity_message = (
            []
            if prev_summary is None
            else [
                {
                    "role": "user",
                    "content": f"Please include these Missing Entities: {','.join(prev_summary.missing)}",
                },
            ]
        )
        new_summary: RewrittenSummary = client.chat.completions.create( # (3)!
            model="gpt-4-0613",
            messages=[
                {
                    "role": "system",
                    "content": """
                You are going to generate an increasingly concise,entity-dense summary of the following article.

                Perform the following two tasks
                - Identify 1-3 informative entities from the following article which is missing from the previous summary
                - Write a new denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities

                Guidelines
                - Make every word count: re-write the previous summary to improve flow and make space for additional entities
                - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
                - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
                - Missing entities can appear anywhere in the new summary
                - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
                """,
                },
                {"role": "user", "content": f"Here is the Article: {article}"},
                {
                    "role": "user",
                    "content": f"Here is the previous summary: {summary_chain[-1]}",
                },
                *missing_entity_message,
            ],
            max_retries=3,
            max_tokens=1000,
            response_model=RewrittenSummary,
        )
        summary_chain.append(new_summary.summary)
        prev_summary = new_summary

    return summary_chain
```

1.  We need to apply a `patch` function on the `OpenAI` client for us to get all
    of the benefits that `Instructor` provides. With a simple `patch`, we can get
    **automatic type coercion of our outputs and automatic retries for invalid outputs**
    out of the box!

2.  We first generate an initial summary. Note here that we explictly ask for a summary that has
    80 words and is lengthy with overly verbose fillers in the system prompt

3.  We slightly modify the original system prompt used in the original paper to perform a rewrite of the summary.
    Using `Instructor`, we also get validation of the generated output with our `field_validator`s that we defined above

This summarization function yields a result which triples the number of entities while mantaining the same number of tokens. We can also see that stylistically, the summary is a lot more natural.

**First Iteration**

> This article discusses the highly-anticipated boxing match between Manny Pacquiao and Floyd Mayweather. The article revolves around Manny Pacquiao's statements about his upcoming fight and his preparations for the same. A portion of the article provides details about the financial stipulations of the match and its significance in the sporting arena. Quotes from Pacquiao illustrating his determination and his battle strategy are highlighted. The tone of the article is largely centered around creating a build-up to the upcoming mega event.

**Final Iteration**

> Manny Pacquiao, the Filipino boxer, anticipates the forthcoming May 2 showdown at the MGM Grand as the fight of his life, against the undefeated American Floyd Mayweather, in a $300m bout. Despite being seen as the underdog in this high-stakes Las Vegas match, Pacquiao is confident, promising a warrior's spirit and assuring the fans who have been awaiting this encounter for a decade, that it will indeed be the biggest sporting spectacle in history worthy of their anticipation

## Part 2) Fine-Tuning

In this section, we'll look into how to fine-tune a GPT 3.5 model so that it is able to perform at an equivalent level as a GPT-4 model. We'll then compare the performance of our model against that of `GPT-4` and `GPT-4-Turbo` to see how it stacks up.

### Creating a Training Set

Let's first segregate our train and test set so that we don't have any sort of contamination - this corresponds to our `train.csv` and `test.csv` in our [Hugging Face Dataset](https://huggingface.co/datasets/ivanleomk/gpt4-chain-of-density). Now, we just need to import the `Instructions` module from the `Instructor` package which allows you to generate a nicely formatted `.jsonl` file to be used for fine-tuning

```py hl_lines="2 9 13-20 25 28"
from typing import List
from chain_of_density import summarize_article #(1)!
import csv
import logging
import instructor
from itertools import islice
from pydantic import BaseModel

client = instructor.patch(OpenAI())

logging.basicConfig(level=logging.INFO)

instructions = instructor.Instructions( #(2)!
    name="Chain Of Density",
    finetune_format="messages",
    # log handler is used to save the data to a file
    # you can imagine saving it to a database or other storage
    # based on your needs!
    log_handlers=[logging.FileHandler("generated.jsonl")],
)

class GeneratedSummary(BaseModel):
    summary: str

@instructions.distil #(3)!
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: List[str] = summarize_article(text)
    return GeneratedSummary(summary=summary_chain[-1]) #(4)!

with open("train.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for index, (article, summary) in enumerate(reader):
        # Run Distillisation to generate the values
        distil_summarization(article)
```

1.  In this example, we're using the summarize_article that we defined up above. We saved it in a local file called `chain_of_density.py`,
    hence the import

2.  We instantiate a `Instruction` object which will help us handle the conversion of our function calls into a valid `.jsonl` file. We also define
    the name of the `.jsonl` file in the `log_handlers` parameter

3.  We add in an `instructions.distil` annotation so that we automatically capture the input and output of the function we'd like to
    fine-tune our model to output

4.  We return a `Pydantic` object which matches the annotation that we use on our function. Note that we must specify a `Pydantic` object to
    be returned when using the `instructions.distil` annotation

!!! warning "Rate Limiting"

    We recommend running this script on a small subset of the dataset first to test you've got everything configured nicely.
    Don't forget to add in rate limiting error handling with `tenacity` and set the `OPENAI_API_KEY` shell environment variable
    before running any subsequent commands

### Creating Fine-Tuning Jobs

Once we run this script, we'll have a new file called `generated.jsonl` in our local repository. Now all that's left is to run the command below to start fine-tuning your first model!

```sh
instructor jobs create-from-file generated.jsonl
```

??? notes "Finetuning Reference"

    Checking out our [Finetuning CLI](/instructor/cli/finetune/) to learn about other hyperparameters that you can tune to improve your model's performance.

Once the job is complete, all we need to do is to then change the annotation in the function call to `distil_summarization` in our original file above to start using our new model.

```py
@instructions.distil(model='gpt-3.5-turbo:finetuned-123', mode="dispatch") #(1)!
def distil_summarization(text: str) -> GeneratedSummary:
    summary_chain: List[str] = summarize_article(text)
    return GeneratedSummary(summary=summary_chain[-1])
```

1. Don't forget to replace this with your new model id. OpenAI identifies fine tuned models with an id of
   ft:gpt-3.5-turbo-0613:personal::<id> under their Fine-tuning tab on their dashboard

With that, you've now got your own fine-tuned model ready to go and serve data in production. We've seen how Instructor can make your life easier, from fine-tuning to distillation.

## Results and Benchmarks

We fine-tuned a total of 3 different models, giving each 20, 50 and 100 samples respectively to see if more data improved the models. We then compared the output of these fine tuned models to GPT-4 and the newly released GPT-4-Turbo to see how they match up.

We'll be comparing these models in three main ways

- Entity Density : This is entities per token, the higher the better for density.
- Latency : Time to last token generated in seconds
- Costs : How much does the entire experiment cost

We used a total of 20 articles as a validation set which our fine tuned models had not seen before. This was the overall performance that we observed.

| Model               | Mean Latency (s) | Mean Entity Count | Mean Entity Density | Tokens |
| ------------------- | ---------------- | ----------------- | ------------------- | ------ |
| GPT-4               | 87.2             | 10.15             | 0.116               | 86.65  |
| GPT-4-Turbo         | 41.1             | 10.05             | 0.116               | 87.25  |
| 3.5 Finetuned (20)  | 2.05             | 10.9              | 0.125               | 87.4   |
| 3.5 Finetuned (50)  | 2.00             | 10.85             | 0.114               | 94.95  |
| 3.5 Finetuned (100) | 2.09             | 10.10             | 0.115               | 88     |

Using the OpenAI Usage Dashboard, we can calculate the cost of generating 20 summaries as seen below.

| Model               | Training Cost ($) | Inference Cost ($) | Tokens Used | Total Cost ($) |
| ------------------- | ----------------- | ------------------ | ----------- | -------------- |
| 3.5 Finetuned (20)  | 0.66              | 0.153              | 43,612      | 0.817          |
| 3.5 Finetuned (50)  | 1.11              | 0.153              | 45,128      | 1.266          |
| 3.5 Finetuned (100) | 2.32              | 0.153              | 44,925      | 2.481          |
| GPT-4 Turbo         | -                 | 1.41               | 265,397     | 1.41           |
| GPT-4               | -                 | 7.63               | 238,290     | 7.69           |

Here, `GPT-4-Turbo` has an approximate inference cost of `$0.07` per summary while our finetuned models have an inference cost of `$0.008` per summary, **which ~ 10x cheaper.**

## Conclusions

Finetuning this iterative method was 20-40x faster while keeping entity density relatively constant. Our token costs also dropped by almost 10x when compared against `GPT-4 Turbo` and by almost 150x when compared against GPT-4, resulting in massive efficiency gains by finetuning and distiling capabilities into specialized models.

We've seen how `Instructor` can make your life easier, from data modeling to distilation and finetuning. If you enjoy the content or want to try out `instructor` check out the [github](https://github.com/jxnl/instructor) and don't forget to give us a star!
