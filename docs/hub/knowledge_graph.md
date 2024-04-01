# Building Knowledge Graphs from Textual Data

In this tutorial, we will explore the process of constructing knowledge graphs from textual data using OpenAI's API and Pydantic. This approach is crucial for efficiently automating the extraction of structured information from unstructured text.

To experiment with this yourself through `instructor hub`, you can obtain the necessary code by executing:

```bash
instructor hub pull --slug knowledge_graph --py > knowledge_graph.py
```

```python
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor


class Node(BaseModel):
    id: int
    label: str
    color: str = "blue"  # Default color set to blue


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"  # Default color for edges


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)


# Patch the OpenAI client to add response_model support
client = instructor.from_openai(OpenAI())


def generate_graph(input_text: str) -> KnowledgeGraph:
    """Generates a knowledge graph from the input text."""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Help me understand the following by describing it as a detailed knowledge graph: {input_text}",
            }
        ],
        response_model=KnowledgeGraph,
    )


if __name__ == "__main__":
    input_text = "Jason is Sarah's friend and he is a doctor"
    graph = generate_graph(input_text)
    print(graph.model_dump_json(indent=2))
    """
    {
      "nodes": [
        {
          "id": 1,
          "label": "Jason",
          "color": "blue"
        },
        {
          "id": 2,
          "label": "Sarah",
          "color": "blue"
        },
        {
          "id": 3,
          "label": "Doctor",
          "color": "blue"
        }
      ],
      "edges": [
        {
          "source": 1,
          "target": 2,
          "label": "friend",
          "color": "black"
        },
        {
          "source": 1,
          "target": 3,
          "label": "is a",
          "color": "black"
        }
      ]
    }
    """
```