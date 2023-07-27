from typing import List

import openai
from pydantic import Field, BaseModel

from openai_function_call import OpenAISchema


class Fact(BaseModel):
    """
    Class representing single statement.
    Each fact has a body and a list of sources.
    If there are multiple facts make sure to break them apart such that each one only uses a set of sources that are relevant to it.
    """

    fact: str = Field(..., description="Body of the sentence, as part of a response")
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
    Class representing a question and its answer as a list of facts each one should have a soruce.
    each sentence contains a body and a list of sources."""

    question: str = Field(..., description="Question that was asked")
    answer: List[Fact] = Field(
        ...,
        description="Body of the answer, each fact should be its seperate object with a body and a list of sources",
    )


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
        temperature=0,
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


question = "What the author's last name?"
context = """
My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.I went to an arts highschool but in university I studied Computational Mathematics and physics.  As part of coop I worked at many companies including Stitchfix, Facebook.  I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
"""


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


answer = ask_ai(question, context)

print("Question:", question)
print()
for fact in answer.answer:
    print("Statement:", fact.fact)
    for span in fact.get_spans(context):
        print("Citation:", highlight(context, span))
    print()
    """
    Question: What did the author do during college?

    Statement: The author studied Computational Mathematics and physics in university.
    Citation: ...s born in China.I went to an arts highschool but <in university I studied Computational Mathematics and physics> . As part of coop I...

    Statement: The author started the Data Science club at the University of Waterloo and was the president of the club for 2 years.
    Citation: ...y companies including Stitchfix, Facebook.I also <started the Data Science club at the University of Waterloo>  and I was the presi...
    Citation: ... club at the University of Waterloo and I was the <president of the club for 2 years> ...
    """
