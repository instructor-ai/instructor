# OpenAI Function Call and Pydantic Integration Module

This Python module provides a powerful and efficient approach to output parsing when interacting with OpenAI's Function Call API. It leverages the data validation capabilities of the Pydantic library to handle output parsing in a more structured and reliable manner. This README will guide you through the installation, usage, and contribution processes of this module.
If you have any feedback, leave an issue or hit me up on [twitter](https://twitter.com/jxnlco). 


## Installation

To get started, clone the repository:

```bash
git clone https://github.com/jxnl/openai_function_call.git
```

Next, install the necessary Python packages from the requirements.txt file:

```bash
pip install -r requirements.txt
```

Note that there's no separate pip install command for this module. Simply copy and paste the module's code into your application.

## Usage

This module simplifies the interaction with the OpenAI API, enabling a more structured and predictable conversation with the AI. Below are examples showcasing the use of function calls and schemas with OpenAI and Pydantic.

### Example 1: Function Calls

```python
@openai_function
def sum(a:int, b:int) -> int:
    """Sum description adds a + b"""
    return a + b

completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        functions=[sum.openai_schema],
        messages=[
            {
                "role": "system",
                "content": "You must use the `sum` function instead of adding yourself.",
            },
            {
                "role": "user",
                "content": "What is 6+3 use the `sum` function",
            },
        ],
    )

result = sum.from_response(completion)
print(result)  # 9
```

### Example 2: Schema Extraction

```python
class UserDetails(OpenAISchema):
    """User Details"""
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    functions=[UserDetails.openai_schema]
    messages=[
        {"role": "system", "content": "I'm going to ask for user details. Use UserDetails to parse this data."},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user_details = UserDetails.from_response(completion)
print(user_details)  # UserDetails(name="John Doe", age=30)
```

## Advanced Usage

### SQL Query Generation with a touch of Security

**Implications:* This script showcases an advanced approach of using OpenAI's GPT-3.5 Turbo model to generate SQL queries based on user requests. Importantly, it also scrutinizes each query for potential security risks, specifically looking for SQL injection attempts. This dual utility brings a new level of automation and safety to data querying, significantly reducing the need for manual query construction and preliminary security assessment.

The SQL class is designed to represent a single SQL query and its parameters, as well as assess the security risk associated with the query. The user-provided input is examined for potential SQL injection attempts or abusive queries, marking these as dangerous when identified. When it comes to generating SQL queries, this script always leans on using query parameters for user-defined inputs, adhering to best security practices.

### Complex Query Planning and Execution

*Implications:* This advanced implementation showcases how complex queries can be decomposed into simpler, dependent sub-queries, allowing the AI to tackle intricate tasks efficiently. This can substantially enhance the depth and accuracy of AI-generated responses, even in situations with multiple unknown variables. Such a tool can be used to drive complex research, provide in-depth answers in a QA system, or support comprehensive data analysis.

The provided Python script demonstrates the construction of a query planner leveraging OpenAI's GPT model. It breaks down a given complex question into multiple smaller, dependent questions or tasks, which are subsequently handled in a specific order (as determined by the dependencies). Asynchronous execution enables the queries to be processed concurrently, and dependencies are taken into account to ensure the proper order of execution. This allows for a highly efficient computation, especially in scenarios with complex, multi-part questions.

### Citation Alignment with QuestionAnswer and Fact Classes

*Implications:* This usage provides a more robust and reliable method for fact extraction and citation. It enhances the reliability of AI outputs, promoting the transparency and traceability of the information. It also presents a method to prevent and minimize the AI's tendency to "hallucinate" or generate unsupported information.

The script employs advanced schema usage to extract and cite specific details from a given context. The Fact class encapsulates each extracted detail, comprising the fact and a list of direct quotes from the context which act as supporting sources. Notably, the citation utilizes an approximate quote produced by the language model and leverages regex with edits to align the citation with an actual substring in the context. This mechanism significantly grounds the fact, ideally minimizing hallucinations. The substring methodology further enables flexible visualization of our citations, shifting from chunk-level references to more precise string-level references.

### Segmenting Single Requests into Multiple Search Queries

*Implications:* The MultiSearch function allows complex tasks to be broken down into simpler, manageable queries, thus enabling parallel processing and potentially improving efficiency and speed. It also opens up possibilities for more complex interactions and more robust responses from the AI.

The MultiSearch function is an advanced feature designed to handle complex scenarios by segmenting a single request into multiple search queries. This ability empowers complex tasks like multitasking and request segmentation, enabling more sophisticated interactions with the OpenAI API.

The Search class defines each search query, consisting of a title, a query, and a search type. The segment function is then used to break the request into multiple search queries, engaging the OpenAI API to segment the request using the MultiSearch class.

### Recursive Data Types in Hierarchical Structures: DirectoryTree and Node Classes

*Implications:* The demonstrated design pattern is crucial for manipulating hierarchical data in a variety of contexts. For example, in computer science, it can facilitate query planning in databases and task management in Directed Acyclic Graph (DAG) execution. Moreover, its utility extends beyond technical applications, proving valuable in organizing complex structures like biological taxonomies or corporate hierarchies.

The DirectoryTree and Node classes demonstrate an advanced usage in handling and manipulating hierarchical data structures with recursive data types. These classes parse a string representation of a filesystem into a structured directory tree, distinguishing between file and folder nodes.

Handling this recursion requires a non-recursive wrapper— in this case, the DirectoryTree class. This approach is due to Pydantic's limitations when dealing with recursive schemas. Wrapping the recursive Node class in a non-recursive DirectoryTree class is a practical workaround for this limitation.

## Contributing

Your contributions are welcome! If you have great examples or find neat patterns, clone the repo and add another example_*.py file. The goal is to find great patterns and cool examples to highlight.

If you encounter any issues or want to provide feedback, you can create an issue in this repository. You can also reach out to me on Twitter at @jxnlco.

## License

This project is licensed under the terms of the MIT license.

For more details, refer to the LICENSE file in the repository.