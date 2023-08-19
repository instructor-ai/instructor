import openai

resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {
            "role": "user",
            "content": "1+1",
        }
    ],
)
print(resp)
