import requests

response = requests.post(
    "http://127.0.0.1:3000/extract",
    json={
        "query": "Alice and Bob are best friends. They are currently 32 and 43 respectively. "
    },
    stream=True,
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        print(str(chunk, encoding="utf-8"), end="\n")
