The general flow for this is: 

1. Decompose the question into subquestions
2. prompt the model by each subquestion and appending it's answer in each subsequent prompt. 

```python
# ---------------
# standard setup and imports
# ---------------

from typing import Dict, Any
from pydantic import BaseModel, Field, create_model
from openai import OpenAI
import instructor


# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.from_openai(OpenAI())


# ---------------
# prompt related schema
# ---------------


# Define the schema for the reasoning object
class ReasoningAnswer(BaseModel):
    reasoning: str
    answer: str


def generate_sub_questions_schema(num_questions: int) -> type:
    """
    Generates a Pydantic schema for subquestions dynamically.

    Args:
        num_questions: The number of subquestions to generate.

    Returns:
        A Pydantic model class with subquestions named 'subQ1', 'subQ2', etc.
    """

    fields = {
        f"subQ{i + 1}": (
            str,
            Field(..., description=f"Subquestion {i + 1}"),
        )  # Required string fields
        for i in range(num_questions)
    }

    return create_model("SubQuestionSchema", **fields)


# Function to decompose the user query into sub-questions
async def decompose_query(query: str, num_sub_questions: int) -> Dict[str, str]:
    decomposition_prompt = (
        f"Decompose the following query into {num_sub_questions} sub-questions: {query}"
    )
    print(f"\nDecomposition Prompt:\n{decomposition_prompt}\n")

    SubQuestionsSchema = generate_sub_questions_schema(num_sub_questions)

    sub_questions_response = client.chat.completions.create(
        messages=[{"role": "user", "content": decomposition_prompt}],
        model="gpt-4o",
        response_model=SubQuestionsSchema,
    )
    sub_questions_data = sub_questions_response.dict()
    print(f"Sub-Questions Response:\n{sub_questions_data}\n")
    return sub_questions_data


# Function to query the language model for each sub-question and build the reasoning object
async def get_reasoning_and_answer(sub_questions: Dict[str, str]) -> ReasoningAnswer:
    reasoning = ""
    answer = ""
    for sub_question in sub_questions.values():
        reasoning_prompt = f"Previous Reasoning: {reasoning}\n Previous Answer: {answer}\n Question: {sub_question}\n Answer:"
        print(f"\nReasoning Prompt:\n{reasoning_prompt}\n")

        reasoning_answer_response = client.chat.completions.create(
            messages=[{"role": "user", "content": reasoning_prompt}],
            model="gpt-4o",
            response_model=ReasoningAnswer,
        )
        reasoning_answer_data = reasoning_answer_response
        print(f"Reasoning: {reasoning_answer_data.reasoning}\nAnswer: {reasoning_answer_data.answer}\n")
        reasoning += reasoning_answer_data.reasoning + " "
        answer = reasoning_answer_data.answer

    return ReasoningAnswer(reasoning=reasoning, answer=answer)


# Main function to handle the user query
async def handle_user_query(query: str, sub_qs: int) -> ReasoningAnswer:
    # Step 1: Decompose the query into sub-questions
    print(f"\nHandling User Query:\n{query}\n")
    sub_questions = await decompose_query(query, sub_qs)

    # Step 2: Get reasoning and answers for each sub-question
    final_reasoning_answer = await get_reasoning_and_answer(sub_questions)
    print(f"\nFinal Reasoning:\n{final_reasoning_answer.reasoning}\n")
    print(f"Final Answer:\n{final_reasoning_answer.answer}\n")

    return final_reasoning_answer


# Example usage
async def main():
    user_query = """
    Question: How do I get Jason's attention with this code I'm writing?
    Additional context: he wrote a tweet asking for some work done in exchange for a bounty reward and some freelance work refs.
    """
    result = await handle_user_query(user_query, 3)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

```

output: 

```markdown



Handling User Query:

    Question: How do I get Jason's attention with this code I'm writing?
    Additional context: he wrote a tweet asking for some work done in exchange for a bounty reward and some freelance work refs.
    


Decomposition Prompt:
Decompose the following query into 3 sub-questions: 
    Question: How do I get Jason's attention with this code I'm writing?
    Additional context: he wrote a tweet asking for some work done in exchange for a bounty reward and some freelance work refs.
    

Sub-Questions Response:
{'subQ1': 'What is the purpose or goal of the code Jason is requesting in his tweet?', 'subQ2': 'What are the expectations or requirements specified by Jason for the code?', 'subQ3': 'How can the code be presented or demonstrated to Jason effectively to get his attention?'}


Reasoning Prompt:
Question: What is the purpose or goal of the code Jason is requesting in his tweet?
Previous Reasoning: 
Answer: 
Provide reasoning and answer for the current question.

Reasoning: Jason's tweet must have requested some specific code functionality or example. The purpose or goal of the code would be dependent on what he is asking for.
Answer: The purpose or goal of the code Jason is requesting in his tweet is dependent on the specific request he made, which could range from demonstrating a feature, fixing a bug, or creating a new function.


Reasoning Prompt:
Question: What are the expectations or requirements specified by Jason for the code?
Previous Reasoning: Jason's tweet must have requested some specific code functionality or example. The purpose or goal of the code would be dependent on what he is asking for. 
Answer: The purpose or goal of the code Jason is requesting in his tweet is dependent on the specific request he made, which could range from demonstrating a feature, fixing a bug, or creating a new function.
Provide reasoning and answer for the current question.

Reasoning Prompt:
Question: How can the code be presented or demonstrated to Jason effectively to get his attention?
Previous Reasoning: Jason's tweet must have requested some specific code functionality or example. The purpose or goal of the code would be dependent on what he is asking for. Understanding Jason's expectations or requirements for the code would depend entirely on the context of his tweet. Without the content of the tweet, we can only speculate on what he is asking for. He could be requesting anything from a demonstration of how a function works to a complete application or a specific fix to an existing problem. 
Answer: The specific expectations or requirements Jason has for the code are dependent on the content and context of his tweet.
Provide reasoning and answer for the current question.

Final Reasoning:
Jason's tweet must have requested some specific code functionality or example. The purpose or goal of the code would be dependent on what he is asking for. Understanding Jason's expectations or requirements for the code would depend entirely on the context of his tweet. Without the content of the tweet, we can only speculate on what he is asking for. He could be requesting anything from a demonstration of how a function works to a complete application or a specific fix to an existing problem. To answer how to present or demonstrate code effectively to Jason and get his attention, we would typically need to understand Jason's background, interests, and the specifics of what he is looking for. However, there are some general best practices that can be applied:

1. **Clarity**: Ensure that the code is clean, well-commented, and easy to understand.
2. **Relevance**: Make sure the code addresses the specific problem or request that Jason has made. Tailor the demonstration to focus on solving his particular issue or displaying the functionality he is interested in.
3. **Visual Aids**: Use visual aids such as diagrams, flowcharts, or even video demonstrations where applicable. For instance, a short video walkthrough of the code might grasp Jason’s attention better than just plain text.
4. **Interactivity**: If possible, provide an interactive demonstration such as a live demo or a sandbox where Jason can play around with the code himself.
5. **Performance**: Highlight the efficiency and performance benefits of the code if this is relevant to the context.
6. **Documentation**: Provide comprehensive documentation that includes not just the 'how', but also the 'why' behind the code. This helps in understanding the thought process and the approach.
7. **Engagement**: Start with a brief introduction that outlines the problem and how the code will address it. Keep it engaging and to the point.

For example, if Jason is looking for a code snippet that demonstrates sorting algorithms, you might present a well-commented Python script that visualizes the sorting process and compares the performance of different algorithms. You could also include a link to an interactive online tool where he can see the algorithms in action. 

Final Answer:
Present the code in a clean, well-commented manner, ensuring it is relevant to Jason's request, and enhance the demonstration with visual aids, interactive elements, and comprehensive documentation to capture his attention effectively.


```