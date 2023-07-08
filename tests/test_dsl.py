from openai_function_call import OpenAISchema, MultiTask
from openai_function_call.dsl import ChatCompletion
from openai_function_call.dsl import messages as m
from openai_function_call.dsl.messages import messages as s


def test_chatcompletion_has_kwargs():
    class Search(OpenAISchema):
        id: int
        query: str

    task = (
        ChatCompletion(name="Acme Inc Email Segmentation", model="gpt3.5-turbo-0613")
        | s.SystemTask(task="Segment emails into search queries")
        | MultiTask(subtask_class=Search)
        | m.TaggedMessage(
            tag="email",
            content="Can you find the video I sent last week and also the post about dogs",
        )
        | m.TipsMessage(
            tips=[
                "When unsure about the correct segmentation, try to think about the task as a whole",
                "If acronyms are used expand them to their full form",
                "Use multiple phrases to describe the same thing",
            ]
        )
        | m.ChainOfThought()
    )
    assert isinstance(task, ChatCompletion)
    assert isinstance(task.kwargs, dict)
