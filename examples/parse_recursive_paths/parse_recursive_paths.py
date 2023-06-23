"""
This script parses a string representation of a filesystem structure into a tree-like directory structure.

The 'Node' class represents a node in this tree, which can be either a file or a folder. Files cannot have 
children, while folders can.

The 'DirectoryTree' class contains a single root folder from which all other files/folders can be reached.
The 'parse_tree_to_filesystem' function uses OpenAI's GPT-3 model to convert a string representation of a 
directory tree into a 'DirectoryTree' object. This object can then be manipulated programmatically as needed,
with methods such as 'print_paths' available for convenience.

Please note: Recursive models currently work if they are wrapped by a non-recursive one. This is why we are
passing a 'DirectoryTree' (which contains a single 'Node') as the function call, not a 'Node' directly. This
is due to a limitation in how Pydantic generates schemas for recursive objects, which creates 
'dict_keys(['$ref', 'definitions'])'. Instead of writing a resolver for such references, we can simply wrap the 
recursive class in a non-recursive one so the function_call class never has a cyclic reference.

Example usage:
>>> root = parse_tree_to_filesystem(
...     '''
...     root
...     ├── folder1
...     │   ├── file1.txt
...     │   └── file2.txt
...     └── folder2
...         ├── file3.txt
...         └── subfolder1
...             └── file4.txt
...     '''
... )
>>> root.print_paths()
# Expected output:
# >>> root                                  NodeType.FOLDER
# >>> root/folder1                          NodeType.FOLDER
# >>> root/folder1/file1.txt                NodeType.FILE
# >>> root/folder1/file2.txt                NodeType.FILE
# >>> root/folder2                          NodeType.FOLDER
# >>> root/folder2/file3.txt                NodeType.FILE
# >>> root/folder2/subfolder1               NodeType.FOLDER
# >>> root/folder2/subfolder1/file4.txt     NodeType.FILE
"""

import enum
from typing import List

import openai
from pydantic import Field
from tenacity import retry, stop_after_attempt

from openai_function_call import OpenAISchema


class NodeType(str, enum.Enum):
    """Enumeration representing the types of nodes in a filesystem."""

    FILE = "file"
    FOLDER = "folder"


class Node(OpenAISchema):
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


class DirectoryTree(OpenAISchema):
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


@retry(stop=stop_after_attempt(3))
def parse_tree_to_filesystem(data: str) -> DirectoryTree:
    """
    Convert a string representing a directory tree into a filesystem structure
    using OpenAI's GPT-3 model.

    Args:
        data (str): The string to convert into a filesystem.

    Returns:
        DirectoryTree: The directory tree representing the filesystem.
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.2,
        functions=[DirectoryTree.openai_schema],
        function_call={"name": DirectoryTree.openai_schema["name"]},
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
    root = DirectoryTree.from_response(completion)
    return root


if __name__ == "__main__":
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
    # >>> root                                  NodeType.FOLDER
    # >>> root/folder1                          NodeType.FOLDER
    # >>> root/folder1/file1.txt                NodeType.FILE
    # >>> root/folder1/file2.txt                NodeType.FILE
    # >>> root/folder2                          NodeType.FOLDER
    # >>> root/folder2/file3.txt                NodeType.FILE
    # >>> root/folder2/subfolder1               NodeType.FOLDER
    # >>> root/folder2/subfolder1/file4.txt     NodeType.FILE
