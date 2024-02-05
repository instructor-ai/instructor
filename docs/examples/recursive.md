# Example: Parsing a Directory Tree

In this example, we will demonstrate how define and use a recursive class definition to convert a string representing a directory tree into a filesystem structure using OpenAI's function call api. We will define the necessary structures using Pydantic, create a function to parse the tree, and provide an example of how to use it.

## Defining the Structures

We will use Pydantic to define the necessary data structures representing the directory tree and its nodes. We have two classes, `Node` and `DirectoryTree`, which are used to model individual nodes and the entire directory tree, respectively.

!!! warning "Flat is better than nested"
While it's easier to model things as nested, returning flat items with dependencies tends to yield better results. For a flat example, check out [planning tasks](planning-tasks.md) where we model a query plan as a dag.

```python
import enum
from typing import List
from pydantic import Field


class NodeType(str, enum.Enum):
    """Enumeration representing the types of nodes in a filesystem."""

    FILE = "file"
    FOLDER = "folder"


class Node(BaseModel):
    """
    Class representing a single node in a filesystem. Can be either a file or a folder.
    Note that a file cannot have children, but a folder can.

    Args:
        name (str): The name of the node.
        children (List[Node]): The list of child nodes (if any).
        node_type (NodeType): The type of the node, either a file or a folder.

    Methods:
        print_paths: Prints the path of the node and its children.
    """

    name: str = Field(..., description="Name of the folder")
    children: List["Node"] = Field(
        default_factory=list,
        description="List of children nodes, only applicable for folders, files cannot have children",
    )
    node_type: NodeType = Field(
        default=NodeType.FILE,
        description="Either a file or folder, use the name to determine which it could be",
    )

    def print_paths(self, parent_path=""):
        """Prints the path of the node and its children."""
        if self.node_type == NodeType.FOLDER:
            path = f"{parent_path}/{self.name}" if parent_path != "" else self.name
            print(path, self.node_type)
            if self.children is not None:
                for child in self.children:
                    child.print_paths(path)
        else:
            print(f"{parent_path}/{self.name}", self.node_type)


class DirectoryTree(BaseModel):
    """
    Container class representing a directory tree.

    Args:
        root (Node): The root node of the tree.

    Methods:
        print_paths: Prints the paths of the root node and its children.
    """

    root: Node = Field(..., description="Root folder of the directory tree")

    def print_paths(self):
        """Prints the paths of the root node and its children."""
        self.root.print_paths()


Node.update_forward_refs()
DirectoryTree.update_forward_refs()
```

The `Node` class represents a single node in the directory tree. It has a name, a list of children nodes (applicable only to folders), and a node type (either a file or a folder). The `print_paths` method can be used to print the path of the node and its children.

The `DirectoryTree` class represents the entire directory tree. It has a single attribute, `root`, which is the root node of the tree. The `print_paths` method can be used to print the paths of the root node and its children.

## Parsing the Tree

We define a function `parse_tree_to_filesystem` to convert a string representing a directory tree into a filesystem structure using OpenAI.

```python
import instructor
from openai import OpenAI

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())


def parse_tree_to_filesystem(data: str) -> DirectoryTree:
    """
    Convert a string representing a directory tree into a filesystem structure
    using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into a filesystem.

    Returns:
        DirectoryTree: The directory tree representing the filesystem.
    """

    return client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        response_model=DirectoryTree,
        messages=[
            {
                "role": "system",
                "content": "You are a perfect file system parsing algorithm. You are given a string representing a directory tree. You must return the correct filesystem structure.",
            },
            {
                "role": "user",
                "content": f"Consider the data below:\n{data} and return the correctly labeled filesystem",
            },
        ],
        max_tokens=1000,
    )
```

The `parse_tree_to_filesystem` function takes a string `data` representing the directory tree and returns a `DirectoryTree` object representing the filesystem structure. It uses the OpenAI Chat API to complete the prompt and extract the directory tree.

## Example Usage

Let's demonstrate how to use the `parse_tree_to_filesystem`

function with an example:

```python
root = parse_tree_to_filesystem(
    """
    root
    ├── folder1
    │   ├── file1.txt
    │   └── file2.txt
    └── folder2
        ├── file3.txt
        └── subfolder1
            └── file4.txt
    """
)
root.print_paths()
```

In this example, we call `parse_tree_to_filesystem` with a string representing a directory tree.

After parsing the string into a `DirectoryTree` object, we call `root.print_paths()` to print the paths of the root node and its children. The output of this example will be:

```python
root                               NodeType.FOLDER
root/folder1                       NodeType.FOLDER
root/folder1/file1.txt             NodeType.FILE
root/folder1/file2.txt             NodeType.FILE
root/folder2                       NodeType.FOLDER
root/folder2/file3.txt             NodeType.FILE
root/folder2/subfolder1            NodeType.FOLDER
root/folder2/subfolder1/file4.txt  NodeType.FILE
```

This demonstrates how to use OpenAI's GPT-3 model to parse a string representing a directory tree and obtain the correct filesystem structure.

I hope this example helps you understand how to leverage OpenAI Function Call for parsing recursive trees. If you have any further questions, feel free to ask!
