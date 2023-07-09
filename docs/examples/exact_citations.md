# Example: Answering Questions with Citations

In this example, we'll demonstrate how to use OpenAI Function Call to ask an AI a question and get back an answer with correct citations. We'll define the necessary data structures using Pydantic and show how to retrieve the citations for each answer.

!!! tips "Motivation"
    When using AI models to answer questions, it's important to provide accurate and reliable information with appropriate citations. By including citations for each statement, we can ensure the information is backed by reliable sources and help readers verify the information themselves.


## Defining the Data Structures

Let's start by defining the data structures required for this task: `Fact` and `QuestionAnswer`.

!!! tip "Prompting as documentation"
    Make sure to include detailed and useful docstrings and fields for your class definitions. Naming becomes very important since they are semantically meaninful in the prompt.

    * `substring_quote` performs better than `quote` since it suggests it should be a substring of the original content.
    * Notice that there are instructions on splitting facts in the docstring which will be used by OpenAI

```python
import openai
from pydantic import Field, BaseModel
from typing import List
from openai_function_call import OpenAISchema


class Fact(BaseModel):
    """
    Each fact has a body and a list of sources.
    If there are multiple facts, make sure to break them apart such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(..., description="Body of the sentence as part of a response")
    substring_quote: List[str] = Field(
        ...,
        description="Each source should be a direct quote from the context, as a substring of the original content",
    )

    def _get_span(self, quote, context, errs=100):
        import regex

        minor = quote
        major = context

        errs_ = 0
        s = regex.search(f"({minor}){{e<={errs_}}}", major)
        while s is None and errs_ <= errs:
            errs_ += 1
            s = regex.search(f"({minor}){{e<={errs_}}}", major)

        if s is not None:
            yield from s.spans()

    def get_spans(self, context):
        for quote in self.substring_quote:
            yield from self._get_span(quote, context)


class QuestionAnswer(OpenAISchema):
    """
    Class representing a question and its answer as a list of facts, where each fact should have a source.
    Each sentence contains a body and a list of sources.
    """

    question: str = Field(..., description="Question that was asked")
    answer: List[Fact] = Field(
        ...,
        description="Body of the answer, each fact should be its separate object with a body and a list of sources",
    )
```

The `Fact` class represents a single statement in the answer. It contains a `fact` attribute for the body of the sentence and a `substring_quote` attribute for the sources, which are direct quotes from the context.

The `QuestionAnswer` class represents a question and its answer. It consists of a `question` attribute for the question asked and a list of `Fact` objects in the `answer` attribute.

!!! tip "Embedding computation"
    While its not thet best idea to get too crazy with adding 100 methods to your class
    colocating some computation is oftentimes useful, here we implement the substring search directly with the `Fact` class.

## Asking AI a Question

To ask the AI a question and get back an answer with citations, we can define a function `ask_ai` that takes a question and context as input and returns a `QuestionAnswer` object.

!!! tips "Prompting Tip: Expert system"
    Expert prompting is a great trick to get results, it can be easily done by saying things like:

    *  you are an world class expert that can correctly ...
    *  you are jeff dean give me a code review ...

```python
def ask_ai(question: str, context: str) -> QuestionAnswer:
    """
    Function to ask AI a question and get back an Answer object.
    but should be updated to use the actual method for making a request to the AI.

    Args:
        question (str): The question to ask the AI.
        context (str): The context for the question.

    Returns:
        Answer: The Answer object.
    """

    # Making a request to the hypothetical 'openai' module
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.2,
        max_tokens=1000,
        functions=[QuestionAnswer.openai_schema],
        function_call={"name": QuestionAnswer.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": f"You are a world class algorithm to answer questions with correct and exact citations. ",
            },
            {"role": "user", "content": f"Answer question using the following context"},
            {"role": "user", "content": f"{context}"},
            {"role": "user", "content": f"Question: {question}"},
            {
                "role": "user",
                "content": f"Tips: Make sure to cite your sources, and use the exact words from the context.",
            },
        ],
    )

    # Creating an Answer object from the completion response
    return QuestionAnswer.from_response(completion)
```

The `ask_ai` function takes a string `question` and a string `context` as input. It makes a completion request to the AI model, providing the question and context as part of the prompt. The resulting completion is then converted into a `QuestionAnswer` object.

## Evaluating an Example

Let's evaluate the example by asking the AI a question and getting back an answer with citations. We'll ask the question "What did the author do during college?" with the given context.

!!! usage "Highlight"
    This just adds some color and captures the citation in `<>`

    ```python
    def highlight(text, span):
        return (
            "..."
            + text[span[0] - 50 : span[0]].replace("\n", "")
            + "\033[91m"
            + "<"
            + text[span[0] : span[1]].replace("\n", "")
            + "> "
            + "\033[0m"
            + text[span[1] : span[1] + 20].replace("\n", "")
            + "..."
        )
    ```

```python
question = "What did the author do during college?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
I went to an arts high school but in university I studied Computational Mathematics and physics. 
As part of coop I worked at many companies including Stitchfix, Facebook.
I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""

answer = ask_ai(question, context)

print("Question:", question)
print()
for fact in answer.answer:
    print("Statement:", fact.fact)
    for span in fact.get_spans(context):
        print("Citation:", highlight(context, span))
    print()
```

In this code snippet, we print the question and iterate over each fact in the answer. For each fact, we print the statement and highlight the corresponding citation in the context using the `highlight` function.

Here is the expected output for the example:

```
Question: What did the author do during college?

Statement: The author studied Computational Mathematics and physics in university.
Citation: ...s born in China.I went to an arts high school but <in university I studied Computational Mathematics and physics> . As part of coop I...

Statement: The author started the Data Science club at the University of Waterloo and was the president of the club for 2 years.
Citation: ...y companies including Stitchfix, Facebook.I also <started the Data Science club at the University of Waterloo>  and I was the presi...
Citation: ... club at the University of Waterloo and I was the <president of the club for 2 years> ...
```

The output includes the question, followed by each statement in the answer with its corresponding citation highlighted in the context.

Feel free to try this code with different questions and contexts to see how the AI responds with accurate citations.