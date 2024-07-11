# Leveraging Local Models for Classifying Private Data

In today's data-driven world, organizations are increasingly concerned about the privacy and security of their sensitive information. Leveraging local models for classifying private data ensures that confidential document queries are handled securely within your own infrastructure. This tutorial will guide you through the process of setting up and using local models to manage and respond to document-related inquiries, providing a robust solution for maintaining data privacy and compliance with security protocols.


## Why Local Models for Classifying Private Data?

local models-cpp provides several key advantages for processing confidential document queries:

1. **Data Privacy**: All processing occurs locally, keeping sensitive information within your secure environment.
2. **Customization**: Local models can be fine-tuned to understand your organization's specific document structures and terminology.
3. **Offline Capability**: Generate responses without internet connectivity, crucial for high-security environments.
4. **Open Source**: Being open-source allows for thorough security audits and customizations.

## Implementing Document Queries with local models

Here's an example of how to implement a system for handling confidential document queries using local models:

```python
import llama_cpp
import instructor
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

# Initialize local models model
local models = llama_cpp.local models(
    model_path="path/to/your/local models/model.gguf",
    n_gpu_layers=-1,
    chat_format="chatml",
    n_ctx=2048,
    verbose=False,
)

# Patch the create function with instructor for structured outputs
create = instructor.patch(
    create=local models.create_chat_completion_openai_v1,
)

# Define query types for document-related inquiries
class QueryType(str, Enum):
    DOCUMENT_CONTENT = "document_content"
    LAST_MODIFIED = "last_modified"
    ACCESS_PERMISSIONS = "access_permissions"
    RELATED_DOCUMENTS = "related_documents"

# Define the structure for query responses
class QueryResponse(BaseModel):
    query_type: QueryType
    response: str
    additional_info: Optional[str] = None

def process_confidential_query(query: str) -> QueryResponse:
    """
    Process a confidential document query using local models.
    
    This function takes a query string, analyzes it using the local models model,
    and returns a structured response about the confidential document.
    
    Args:
        query (str): The user's document-related query.
    
    Returns:
        QueryResponse: A structured response containing the query type,
                       response text, confidence score, and any additional info.
    """
    prompt = f"""Analyze the following confidential document query and provide an appropriate response:
    Query: {query}
    
    Determine the type of query (document content, last modified, access permissions, or related documents),
    provide a response, and include a confidence score and any additional relevant information.
    Remember, you're handling confidential data, so be cautious about specific details.
    """

    result = create(
        response_model=QueryResponse,
        messages=[
            {"role": "system", "content": "You are a secure AI assistant trained to handle confidential document queries."},
            {"role": "user", "content": prompt},
        ],
    )
    return result

# Sample confidential document queries
confidential_queries = [
    "What are the key findings in the Q4 financial report?",
    "Who last accessed the merger proposal document?",
    "What are the access permissions for the new product roadmap?",
    "Are there any documents related to Project X's budget forecast?",
    "When was the board meeting minutes document last updated?",
]

print("#> Processing confidential document queries...\n")

# Process each query and print the results
for i, query in enumerate(confidential_queries, 1):
    response = process_confidential_query(query)
    print(f"#> Query {i}: {query}")
    print(f"#> Query Type: {response.query_type}")
    print(f"#> Response: {response.response}")
    if response.additional_info:
        print(f"#> Additional Info: {response.additional_info}")
    print("#>")
```

This code demonstrates how local models can be used to handle various types of confidential document-related queries locally, ensuring data privacy and security.

## Example Outputs

Here's what the output might look like:

```
#> Processing confidential document queries...
#>
#> Query 1: What are the key findings in the Q4 financial report?
#> Query Type: document_content
#> Response: I cannot disclose specific financial details, but the Q4 financial report typically includes information on revenue, expenses, and profit margins for the fourth quarter. For accurate and authorized information, please consult the official report or speak with the finance department.
#> Additional Info: Financial reports often compare results to previous quarters and provide future projections.
#>
#> Query 2: Who last accessed the merger proposal document?
#> Query Type: last_modified
#> Response: For security reasons, I don't have direct access to document access logs. To find out who last accessed the merger proposal document, please consult your document management system's audit logs or contact the IT security team.
#>
#> Query 3: What are the access permissions for the new product roadmap?
#> Query Type: access_permissions
#> Response: I don't have real-time access to document permissions. Access to the new product roadmap is likely restricted to key stakeholders. For specific access permissions, please check with the document owner or the IT department.
#> Additional Info: Product roadmaps often have tiered access levels based on roles and need-to-know basis.
#>
#> Query 4: Are there any documents related to Project X's budget forecast?
#> Query Type: related_documents
#> Response: I don't have access to the full document repository. There may be documents related to Project X's budget forecast, but for security reasons, I can't confirm their existence or content. Please consult your project management tool or speak with the project manager for accurate information.
#>
#> Query 5: When was the board meeting minutes document last updated?
#> Query Type: last_modified
#> Response: I don't have direct access to document modification timestamps. To find out when the board meeting minutes were last updated, please check the document properties in your document management system or consult with the board secretary.
```

## Benefits in Real-World Scenarios

Using local models for local confidential document queries offers several advantages:

1. **Enhanced Security**: Sensitive document information never leaves your local environment.
2. **Customized Responses**: local models can be fine-tuned to understand your organization's document types and security protocols.
3. **Versatility**: Handle various query types while maintaining strict confidentiality.
4. **Scalability**: Process a large volume of queries without compromising on security or speed.

## Conclusion

 provides a robust solution for organizations needing to handle confidential document queries locally. By processing these queries on your own hardware, you can leverage advanced AI capabilities while maintaining the highest standards of data privacy and security.

To implement this effectively, focus on:
1. Choosing the right local models model size for your computational resources.
2. Fine-tuning the model on your organization's document types and common queries.
3. Implementing strict access controls around the local models system itself.
4. Regularly updating and auditing the system to ensure it aligns with your security policies.

With these considerations in mind, you can provide efficient, AI-powered assistance for document queries while keeping your sensitive information secure and compliant with data protection regulations.