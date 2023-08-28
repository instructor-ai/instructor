# Legal Document Entity Resolution

This example demonstrates how to use an entity resolution system to extract and resolve entities from a legal document. The system leverages OpenAI's GPT-4 language model to achieve this task. The primary purpose of this example is to showcase the capabilities of the entity resolution system in a simple and illustrative manner.

## Overview
The entity resolution system processes a given legal document and identifies key entities such as parties, dates, terms, and clauses. It then resolves relevant information to provide a structured output. This example uses a Python script to interact with the system and demonstrates the process with a sample legal contract.

## How to Use

* **Input Document:** Provide the legal document you want to analyze. The document should include relevant legal terms, dates, parties' names, and other pertinent information.

* **Entity Extraction:** The system employs the GPT-4 model to extract entities from the input document.

* **Entity Resolution:** Extracted entities are resolved to their absolute values when applicable. For instance, relative date phrases are converted to specific dates.

* **Dependency Handling:** The system identifies dependencies between entities. If one entity's resolution depends on another's, it ensures proper order of resolution.

## Limitations

The context window is the biggest limitation of the size of document, but I imagine a system where you stream chunks of the document into a model, that acculimates the entities in some state and formats a simple version back into the prompt (id, name, absolute_resolved_value) and the output emits only 'new' entities, thinking of it as a acculilating the object.