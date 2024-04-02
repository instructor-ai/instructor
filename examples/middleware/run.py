import instructor
import openai
from openai.types.chat import ChatCompletionMessageParam
from typing import List
from pydantic import BaseModel


class PrintLastUserMessage(instructor.MessageMiddleware):
    log: bool = False

    def __call__(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        if self.log:
            import pprint

            pprint.pprint({"messages": messages})
        return messages


@instructor.messages_middleware
def dumb_rag(messages):
    # TODO: use RAG to generate a response
    # TODO: add the response to the messages
    return messages + [
        {
            "role": "user",
            "content": "Search retrieved: 'Jason is 20 years old'",
        }
    ]


try:

    @instructor.messages_middleware
    def dumb_rag(x):
        return x

except ValueError as e:
    print("Correctly caught exception", e)


class User(BaseModel):
    age: int
    name: str


client = (
    instructor.from_openai(openai.OpenAI())
    .with_middleware(dumb_rag)
    .with_middleware(PrintLastUserMessage(log=True))  # can be called directly
)


user = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {
            "role": "user",
            "content": "How old is jason?",
        }
    ],
    response_model=User,
)

print(user)
# {'messages': [{'content': 'How old is jason?', 'role': 'user'},
#               {'content': "Search retrieved: 'Jason is 20 years old'",
#                'role': 'user'}]}
# {'age': 20, 'name': 'jason'}
