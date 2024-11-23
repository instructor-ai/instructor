#!/usr/bin/env python
# coding: utf-8

# # Validators

# Instead of framing "self-critique" or "self-reflection" in AI as new concepts, we can view them as validation errors with clear error messages that the system can use to self correct.
# 
# Pydantic offers an customizable and expressive validation framework for Python. Instructor leverages Pydantic's validation framework to provide a uniform developer experience for both code-based and LLM-based validation, as well as a reasking mechanism for correcting LLM outputs based on validation errors. To learn more check out the Pydantic [docs](https://docs.pydantic.dev/latest/) on validators.
# 
# Note: For the majority of this notebook we won't be calling openai, just using validators to see how we can control the validation of the objects.

# Validators will enable us to control outputs by defining a function like so:
# 
# 
# ```python
# def validation_function(value):
#     if condition(value):
#         raise ValueError("Value is not valid")
#     return mutation(value)
# ```
# 
# Before we get started lets go over the general shape of a validator:

# In[61]:


from pydantic import BaseModel
from typing import Annotated
from pydantic import AfterValidator

def name_must_contain_space(v: str) -> str:
    if " " not in v:
        raise ValueError("Name must contain a space.")
    return v.lower()

class UserDetail(BaseModel):
    age: int
    name: Annotated[str, AfterValidator(name_must_contain_space)]

person = UserDetail(age=29, name="Jason")


# **Validation Applications**
# 
# Validators are essential in tackling the unpredictabile nature of LLMs.
# 
# Straightforward examples include:
# 
# * Flagging outputs containing blacklisted words.
# * Identifying outputs with tones like racism or violence.
# 
# For more complex tasks:
# 
# * Ensuring citations directly come from provided content.
# * Checking that the model's responses align with given context.
# * Validating the syntax of SQL queries before execution.

# ## Setup and Dependencies

# Using the [instructor](https://github.com/jxnl/instructor) library, we streamline the integration of these validators. `instructor` manages the parsing and validation of outputs and automates retries for compliant responses. This simplifies the process for developers to implement new validation logic, minimizing extra overhead.

# To use instructor in our api calls, we just need to patch the openai client:

# In[5]:


import instructor 
from openai import OpenAI

client = instructor.patch(OpenAI())


# ## Software 2.0: Rule-based validators

# Deterministic validation, characterized by its rule-based logic, ensures consistent outcomes for the same input. Let's explore how we can apply this concept through some examples.

# ### Flagging bad keywords

# To begin with, we aim to prevent engagement in topics involving explicit violence.

# We will define a blacklist of violent words that cannot be mentioned in any messages:

# In[63]:


blacklist = {
    "rob",
    "steal",
    "hurt",
    "kill",
    "attack",
}


# To validate if the message contains a blacklisted word we will use a [field_validator](https://jxnl.github.io/instructor/blog/2023/10/23/good-llm-validation-is-just-good-validation/#using-field_validator-decorator) over the 'message' field:

# In[64]:


from pydantic import BaseModel, field_validator
from pydantic.fields import Field

class Response(BaseModel):
    message: str

    @field_validator('message')
    def message_cannot_have_blacklisted_words(cls, v: str) -> str:
        for word in v.split(): 
            if word.lower() in blacklist:
                raise ValueError(f"`{word}` was found in the message `{v}`")
        return v

Response(message="I will hurt him")


# ### Flagging using OpenAI Moderation

# To enhance our validation measures, we'll extend the scope to flag any answer that contains hateful content, harassment, or similar issues. OpenAI offers a moderation endpoint that addresses these concerns, and it's freely available when using OpenAI models.

# With the `instructor` library, this is just one function edit away:

# In[1]:


from typing import Annotated
from pydantic.functional_validators import AfterValidator


# In[6]:


from instructor import openai_moderation

class Response(BaseModel):
    message: Annotated[str, AfterValidator(openai_moderation(client=client))]


# Now we have a more comprehensive flagging for violence and we can outsource the moderation of our messages.

# In[7]:


Response(message="I want to make them suffer the consequences")


# And as an extra, we get flagging for other topics like religion, race etc.

# In[26]:


Response(message="I will mock their religion")


# ### Filtering very long messages

# In addition to content-based flags, we can also set criteria based on other aspects of the input text. For instance, to maintain user engagement, we might want to prevent the assistant from returning excessively long texts. 
# 
# Here, noticed that `Field` has built-in validators for `min_length` and `max_length`. to learn more checkout [Field Contraints](https://docs.pydantic.dev/latest/concepts/fields)

# In[68]:


class AssistantMessage(BaseModel):
    message: str = Field(..., max_length=100)


# In[69]:


AssistantMessage(message="Certainly! Lorem ipsum is a placeholder text commonly used in the printing and typesetting industry. Here's a sample of Lorem ipsum text: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod velit vel tellus tempor, non viverra eros iaculis. Sed vel nisl nec mauris bibendum tincidunt. Vestibulum sed libero euismod, eleifend tellus id, laoreet elit. Donec auctor arcu ac mi feugiat, vel lobortis justo efficitur. Fusce vel odio vitae justo varius dignissim. Integer sollicitudin mi a justo bibendum ultrices. Quisque id nisl a lectus venenatis luctus. Please note that Lorem ipsum text is a nonsensical Latin-like text used as a placeholder for content, and it has no specific meaning. It's often used in design and publishing to demonstrate the visual aspects of a document without focusing on the actual content.")


# ### Avoiding hallucination with citations

# When incorporating external knowledge bases, it's crucial to ensure that the agent uses the provided context accurately and doesn't fabricate responses. Validators can be effectively used for this purpose. We can illustrate this with an example where we validate that a provided citation is actually included in the referenced text chunk:

# In[70]:


from pydantic import ValidationInfo

class AnswerWithCitation(BaseModel):
    answer: str
    citation: str

    @field_validator('citation')
    @classmethod
    def citation_exists(cls, v: str, info: ValidationInfo): 
        context = info.context
        if context:
            context = context.get('text_chunk')
            if v not in context:
                raise ValueError(f"Citation `{v}` not found in text")
        return v


# Here we assume that there is a "text_chunk" field that contains the text that the model is supposed to use as context. We then use the `field_validator` decorator to define a validator that checks if the citation is included in the text chunk. If it's not, we raise a `ValueError` with a message that will be returned to the user.

# In[71]:


AnswerWithCitation.model_validate(
    {
        "answer": "Blueberries are packed with protein", 
        "citation": "Blueberries contain high levels of protein"
    },
    context={"text_chunk": "Blueberries are very rich in antioxidants"}, 
)


# ## Software 3.0: Probabilistic validators

# For scenarios requiring more nuanced validation than rule-based methods, we use probabilistic validation. This approach incorporates LLMs into the validation workflow for a sophisticated assessment of outputs.
# 
# The `instructor` library offers the `llm_validator` utility for this purpose. By specifying the desired directive, we can use LLMs for complex validation tasks. Let's explore some intriguing use cases enabled by LLMs.
# 
# ### Keeping an agent on topic
# 
# When creating an agent focused on health improvement, providing answers and daily practice suggestions, it's crucial to ensure strict adherence to health-related topics. This is important because the knowledge base is limited to health topics, and veering off-topic could result in fabricated responses.
# 
# To achieve this focus, we'll follow a similar process as before, but with an important addition: integrating an LLM into our validator.

# This LLM will be tasked with determining whether the agent's responses are exclusively related to health topics. For this, we will use the `llm_validator` from `instructor` like so:

# In[73]:


from instructor import llm_validator

class AssistantMessage(BaseModel):
    message: Annotated[str,
                       AfterValidator(
                           llm_validator("don't talk about any other topic except health best practices and topics",
                                         client=client))]

AssistantMessage(message="I would suggest you to visit Sicily as they say it is very nice in winter.")


# Important that for these examples we're not waiting for the messages, to get this message we would need to call the openai with `response_model=AssistantMessage`.

# ### Validating agent thinking with CoT

# Using probabilistic validation, we can also assess the agent's reasoning process to ensure it's logical before providing a response. With [chain of thought](https://learnprompting.org/docs/intermediate/chain_of_thought) prompting, the model is expected to think in steps and arrive at an answer following its logical progression. If there are errors in this logic, the final response may be incorrect.
#
# Here we will use Pydantic's field_validator with mode="after" which allows us to validate a field while having access to
# other model fields through the ValidationInfo object. This is particularly useful for validating relationships between fields,
# like ensuring the answer follows from the chain of thought.
#
# To make this easier we'll make a simple validation class that we can reuse for all our validation:

# In[74]:


from typing import Optional

class Validation(BaseModel):
    is_valid: bool = Field(..., description="Whether the value is valid based on the rules")
    error_message: Optional[str] = Field(..., description="The error message if the value is not valid, to be used for re-asking the model")


# The function we will call will integrate an LLM and will ask it to determine whether the answer the model provided follows from the chain of thought: 

# In[75]:


def validate_chain_of_thought(values):
    chain_of_thought = values["chain_of_thought"]
    answer = values["answer"]
    resp = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a validator. Determine if the value follows from the statement. If it is not, explain why.",
            },
            {
                "role": "user",
                "content": f"Verify that `{answer}` follows the chain of thought: {chain_of_thought}",
            },
        ],
        response_model=Validation,
    )
    if not resp.is_valid:
        raise ValueError(resp.error_message)
    return values


# The use of the 'before' argument in this context is significant. It means that the validator will receive the complete dictionary of inputs in their raw form, before any parsing by Pydantic.

# In[76]:


from typing import Any
from pydantic import ValidationInfo

class AIResponse(BaseModel):
    chain_of_thought: str
    answer: str

    @field_validator("answer", mode="after")
    @classmethod
    def chain_of_thought_makes_sense(cls, answer: str, info: ValidationInfo) -> str:
        # Get the chain_of_thought from the model data
        chain_of_thought = info.data.get("chain_of_thought")
        # Validate using our helper function
        validate_chain_of_thought({
            "chain_of_thought": chain_of_thought,
            "answer": answer
        })
        return answer


# In[77]:


AIResponse(chain_of_thought="The user suffers from diabetes.", answer="The user has a broken leg.")


# ## Reasking with validators
# 
# For most of these examples all we've done we've mostly only defined the validation logic.
# 
# We'eve covered field validators and model validators and even used LLMs to validate our outputs. But we haven't actually used the validators to reask the model! One of the most powerful features of `instructor` is that it will automatically reask the model when it receives a validation error. This means that we can use the same validation logic for both code-based and LLM-based validation.
# 
# This also means that our 'prompt' is not only the prompt we send, but the code that runs the validator, and the error message we send back to the model.

# Integrating these validation examples with the OpenAI API is streamlined using `instructor`. After patching the OpenAI client with `instructor`, you simply need to specify a `response_model` for your requests. This setup ensures that all the validation processes occur automatically.
# 
# To enable reasking you can set a maximum number of retries. When calling the OpenAI client, the system can re-attempt to generate a correct answer. It does this by resending the original query along with feedback on why the previous response was rejected, guiding the LLM towards a more accurate answer in subsequent attempts.

# In[79]:


class QuestionAnswer(BaseModel):
    question: str
    answer: str

question = "What is the meaning of life?"
context = "The according to the devil the meaning of life is a life of sin and debauchery."


resp = client.chat.completions.create(
    model="gpt-4-1106-preview",
    response_model=QuestionAnswer,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: `{context}`\n\nAnswer the following question: `{question}`",
        },
    ],
)

resp.answer


# In[80]:


from pydantic import BeforeValidator

class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", client=client)
        ),
    ]

resp = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=QuestionAnswer,
    max_retries=2,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: `{context}`\n\nAnswer the following question: `{question}`",
        },
    ],
)

resp.answer


# # Conclusion

# This guide explains how to use deterministic and probabilistic validation techniques with Large Language Models (LLMs). We discussed using an instructor to establish validation processes for content filtering, context relevance maintenance, and model reasoning verification. These methods enhance the performance of LLMs across different tasks.
# 
# For those interested in further exploration, here's a to-do list:
# 
# 1. **SQL Syntax Checker**: Create a validator to check the syntax of SQL queries before executing them.
# 2. **Context-Based Response Validation**: Design a method to flag responses based on the model's own knowledge rather than the provided context.
# 3. **PII Detection**: Implement a mechanism to identify and handle Personally Identifiable Information in responses while prioritizing user privacy.
# 4. **Targeted Rule-Based Filtering**: Develop filters to remove specific content types, such as responses mentioning named entities.
# 
# Completing these tasks will enable users to acquire practical skills in improving LLMs through advanced validation methods.
