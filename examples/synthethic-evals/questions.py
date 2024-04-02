from pydantic import BaseModel, Field
import instructor
from openai import AsyncOpenAI
from asyncio import run
from typing import List
from tqdm.asyncio import tqdm_asyncio as asyncio
from rich.table import Table
from rich.console import Console

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


async def generate_questions(chunks: List[str]):
    coros = [generate_question_answer_pair(chunk) for chunk in chunks]
    return await asyncio.gather(*coros)


if __name__ == "__main__":
    chunks = [
        "The companies at the Google end of the continuum are called startups when they're young. The reason I know about them is that my wife Jessica and I started something called Y Combinator that is basically a startup factory. Since 2005, Y Combinator has funded over 4000 startups. So we know exactly what you need to start a startup, because we've helped people do it for the last 19 years.",
        "All you can know when you start working on a startup is that it seems worth pursuing. You can't know whether it will turn into a company worth billions or one that goes out of business. So when I say I'm going to tell you how to start Google, I mean I'm going to tell you how to get to the point where you can start a company that has as much chance of being Google as Google had of being Google.",
        "Those of you who are taking computer science classes in school may at this point be thinking, ok, we've got this sorted. We're already being taught all about programming. But sorry, this is not enough. You have to be working on your own projects, not just learning stuff in classes. You can do well in computer science classes without ever really learning to program. In fact you can graduate with a degree in computer science from a top university and still not be any good at programming. That's why tech companies all make you take a coding test before they'll hire you, regardless of where you went to university or how well you did there. They know grades and exam results prove nothing.",
    ]

    questions: List[QuestionAnswerPair] = run(generate_questions(chunks))

    table = Table(title="Questions and Sources", show_lines=True)
    table.add_column(
        "Question", style="magenta", justify="left", no_wrap=False, max_width=30
    )
    table.add_column(
        "Original Source", style="cyan", justify="left", no_wrap=False, max_width=50
    )

    for question, chunk in zip(questions, chunks):
        table.add_row(question.question, chunk)

    console = Console()
    console.print(table, justify="center")
