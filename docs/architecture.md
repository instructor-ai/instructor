---
title: Instructor Architecture
description: Understand how Instructor works under the hood with various LLM providers
---

# Instructor Architecture

This document explains how Instructor works internally and how it integrates with different LLM providers.

## Core Components

Instructor consists of several key components that work together:

```mermaid
graph TD
    User[User Code] --> PydanticModel[Pydantic Model]
    User --> ClientInit[Initialize Provider Client]
    ClientInit --> PatchedClient[Patched LLM Client]
    PydanticModel --> SchemaConverter[Schema Converter]
    SchemaConverter --> ProviderAdapter[Provider-Specific Adapter]
    PatchedClient --> APIRequest[API Request]
    ProviderAdapter --> APIRequest
    APIRequest --> ResponseParser[Response Parser]
    ResponseParser --> Validator[Pydantic Validator]
    Validator -- Valid --> StructuredOutput[Structured Output]
    Validator -- Invalid --> RetryMechanism[Retry Mechanism]
    RetryMechanism --> APIRequest
```

### Components Explained

1. **Pydantic Model**: Defines the structure of the data you want to extract
2. **Schema Converter**: Transforms Pydantic models into schemas understood by LLMs
3. **Provider-Specific Adapter**: Formats requests for specific LLM providers
4. **Patched LLM Client**: Augments provider clients with structured output capabilities
5. **Response Parser**: Extracts structured data from LLM responses
6. **Validator**: Validates responses against the Pydantic model
7. **Retry Mechanism**: Automatically retries with feedback when validation fails

## Request Flow

Here's how a typical Instructor request flows through the system:

```mermaid
sequenceDiagram
    participant User
    participant Instructor
    participant ProviderClient
    participant LLM

    User->>Instructor: create(response_model=Model, messages=[...])
    Instructor->>Instructor: Convert Pydantic model to schema
    Instructor->>Instructor: Format request for provider
    Instructor->>ProviderClient: Forward request with tools/functions
    ProviderClient->>LLM: Make API request
    LLM->>ProviderClient: Return response
    ProviderClient->>Instructor: Return response
    Instructor->>Instructor: Parse structured data
    Instructor->>Instructor: Validate against model
    
    alt Validation succeeds
        Instructor->>User: Return validated Pydantic object
    else Validation fails
        Instructor->>Instructor: Generate error feedback
        Instructor->>ProviderClient: Retry with feedback
        ProviderClient->>LLM: Make new request
        LLM->>ProviderClient: Return new response
        ProviderClient->>Instructor: Return new response
        Instructor->>Instructor: Parse and validate again
        Instructor->>User: Return validated Pydantic object
    end
```

This architecture enables Instructor to provide a consistent interface across different LLM providers while handling their specific implementation details behind the scenes.