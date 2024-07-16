# Batch Embeddings

if you're running any embedding jobs that aren't time sensitive, you can now use the OpenAI Batch API with instructor to save 50% of inference costs.

!!! info "Batch Job Limits"

    Note that there is a limit of 50,000 input items for a single embedding batch job at the moment.

Let's see a quick example on how to do so.

## Creating the `.jsonl` file

We first need to create a valid `.jsonl` file that can be uploaded as a batch job. We can do so with the `BatchJob` class.

```python
from instructor.batch import BatchJob

examples = [
    "this is a test",
    "this is another test",
    "this is a third test",
]

BatchJob.embed_list(
    examples,
    model="text-embedding-3-small",
    file_path="./embeddings.jsonl",
    dimensions=300, # This is an optional parameter, if not set, it will default to the full dimensionality of the model
)
```

This in turn creates a `.jsonl` file called `embeddings.jsonl` in your local directory. We can then upload this to OpenAI by using the command

```
instructor batch create-from-file --file-path ./embeddings.jsonl --endpoint '/v1/embeddings'
```

This will then create an embedding job which you can monitor using our cli using the command

```
instructor batch list
```

| Batch ID                       | Created At          | Status    | Failed | Completed | Total |
| ------------------------------ | ------------------- | --------- | ------ | --------- | ----- |
| batch_igaa2j9VBVw2ZWwdTFzurdlb | 2024-07-16 17:12:10 | completed | 0      | 3         | 3     |

Make sure to copy the full `batch_id` of the job that you want to download the results for and then use the following commmand to download the finished result.

```
instructor batch download-file --batch-id batch_igaa2j9VBVw2ZWwdTFzurdlb --download-file-path ./output-embed.jsonl
```

## Extracting embeddings

Extracting the embeddings from the output file is easy, just use the `read_embeddings_from_file` method

```python
BatchJob.embed_list(
    examples,
    model="text-embedding-3-small",
    file_path="./embeddings.jsonl",
    dimensions=300,
)

embeddings = BatchJob.parse_embeddings_from_file("./output-embed.jsonl")
```
