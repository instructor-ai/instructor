# Philosophy

The philosophy behind this library is to provide a **lightweight** and **flexible** approach to leveraging language models (LLMs) to do **structured output without imposing unnecessary dependencies or abstractions.**

By treating LLMs as just another function that returns a typed object, this library aims to remove the perceived complexity and make working with LLMs more approachable. It provides a flexible foundation for incorporating LLMs into your projects while allowing you to leverage the full power of Python to write your own code.

1. Define a Schema `#!python class StructuredData(OpenAISchema):`
2. Encapsulate all your LLM logic into a function `#!python def extract(a) -> StructuredData:` 
3. Define typed computations against your data with `#!python def compute(data: StructuredData):`


Here are some key points to understand:

* **Minimal Installation:** This library is designed to be lightweight. If you prefer not to install the library and its two dependencies, you can simply extract the `function_calls.py` file from the code and incorporate it directly into your project. Own it. It is a single script. 

* **Code as prompts:** With both the DSL and the structured extraction we don't make a distinction between a code vs a `prompt template`. We believe the prompts that go into a LLM should be constructed and collocated with the code we need to execute. Prompts are created via docstrings, descriptions and functions that construct messages.

* **Writing Prompts:** The library also includes an experimental prompt pipeline api. The DSL is a thin wrapper that aims to improve code readability by adding light abstraction around templates as messages. It provides a slightly more intuitive syntax for working with LLMs, making the prompting easier to read.

    * **No Abstractions for Retrieval or Execution:** The library does not impose any abstractions for retrieval or execution. Python code is considered to be great glue code, and there is no need to force the use of additional abstractions. You have the freedom to use any retrieval or execution mechanism that suits your needs.

    !!! example "Roll your own"
        If you want to do retrival simply make a class that extracts a query and implement a method that calls to get the data!

        ```python
        import ...

        class Search(OpenAISchema):
            "Extract the search query from a question a query"
            query: str 

            def execute(self):
                return vectordb.query(self.query).to_string()

        req = (
            ChatCompletion("Query Extraction")
            | TaggedMessage(content="can you tell me who is Jason Liu?", tag="query")
            | Search
        ) 
        query = req.create() # Search(query="Jason Liu")
        query.execute()      # "Jason Liu has a twitter account @jxnlco"
        ```

    * **Message as the Building Block:** In the DSL, the fundamental building block is a message or an OpenAI schema call. Retrieval should be implemented as code that returns a message to be used in retrieval-augmented generation. This approach allows for flexibility and customization in handling different types of inputs and outputs.

    !!! example "Roll your own, again"
        If you want to do augmentation, you can simple make a function that returns the data in a message and `|` it back into the completion.

        ```python
        import ...

        class Response(OpenAISchema):
            "Question answer pairs"
            question: str
            answer: str


        def augment(query):
            return TaggedMessage(content=vectordb.query(self.query).to_string(), tag="data")


        query = "does jason have social media?"
        req = (
            ChatCompletion("Q/A System")
            | augment(query=query)
            | UserMessage(f"Use the data to answer the question: {query}")
            | Response
        )
        response = req.create()
        # Response(
        #   question="does jason have social media?", 
        #   answer="yes, his twitter is @jxnlco
        # )
        ```

Please note that the library is designed to be adaptable and open-ended, allowing you to customize and extend its functionality based on your specific requirements.

If you have any further questions or ideas hit me up on [twitter](https://twitter.com/jxnlco)
