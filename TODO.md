# Refactoring TODO List

## Phase 1: Create Provider-Specific Directory Structure
- [x] Create utils directory
- [x] Create utils/__init__.py
- [x] Create core.py with generic utilities
- [x] Create providers.py with Provider enum and detection
- [x] Create anthropic.py with Anthropic-specific utilities
- [x] Create openai.py with OpenAI-specific utilities
- [x] Create google.py with Google-specific utilities
- [x] Create cohere.py with Cohere-specific utilities
- [x] Create mistral.py with Mistral-specific utilities
- [x] Create bedrock.py with AWS Bedrock utilities
- [x] Create fireworks.py with Fireworks utilities
- [x] Create cerebras.py with Cerebras utilities
- [x] Create writer.py with Writer utilities
- [x] Create perplexity.py with Perplexity utilities

## Phase 2: Extract and Move Functions

### From utils.py to core.py:
- [ ] Move extract_json_from_codeblock
- [ ] Move extract_json_from_stream
- [ ] Move extract_json_from_stream_async
- [ ] Move update_total_usage
- [ ] Move dump_message
- [ ] Move is_async
- [ ] Move merge_consecutive_messages
- [ ] Move classproperty
- [ ] Move get_message_content
- [ ] Move disable_pydantic_error_url
- [ ] Move is_typed_dict
- [ ] Move prepare_response_model

### From utils.py to providers.py:
- [ ] Move Provider enum
- [ ] Move get_provider function

### From utils.py to anthropic.py:
- [ ] Move SystemMessage TypedDict
- [ ] Move combine_system_messages
- [ ] Move extract_system_messages

### From utils.py to google.py:
- [ ] Move transform_to_gemini_prompt
- [ ] Move verify_no_unions
- [ ] Move map_to_gemini_function_schema
- [ ] Move update_genai_kwargs
- [ ] Move update_gemini_kwargs
- [ ] Move extract_genai_system_message
- [ ] Move convert_to_genai_messages

### From reask.py to provider modules:
- [ ] Move reask_anthropic_tools to anthropic.py
- [ ] Move reask_anthropic_json to anthropic.py
- [ ] Move reask_gemini_tools to google.py
- [ ] Move reask_gemini_json to google.py
- [ ] Move reask_vertexai_tools to google.py
- [ ] Move reask_vertexai_json to google.py
- [ ] Move reask_genai_tools to google.py
- [ ] Move reask_genai_structured_outputs to google.py
- [ ] Move reask_cohere_tools to cohere.py
- [ ] Move reask_mistral_tools to mistral.py
- [ ] Move reask_mistral_structured_outputs to mistral.py
- [ ] Move reask_bedrock_json to bedrock.py
- [ ] Move reask_fireworks_tools to fireworks.py
- [ ] Move reask_fireworks_json to fireworks.py
- [ ] Move reask_cerebras_tools to cerebras.py
- [ ] Move reask_writer_tools to writer.py
- [ ] Move reask_writer_json to writer.py
- [ ] Move reask_perplexity_json to perplexity.py
- [ ] Move reask_tools to openai.py
- [ ] Move reask_responses_tools to openai.py
- [ ] Move reask_md_json to openai.py
- [ ] Move reask_default to openai.py

### From process_response.py to provider modules:
- [ ] Move handle_anthropic_tools to anthropic.py
- [ ] Move handle_anthropic_json to anthropic.py
- [ ] Move handle_anthropic_reasoning_tools to anthropic.py
- [ ] Move handle_anthropic_parallel_tools to anthropic.py
- [ ] Move handle_gemini_tools to google.py
- [ ] Move handle_gemini_json to google.py
- [ ] Move handle_vertexai_tools to google.py
- [ ] Move handle_vertexai_json to google.py
- [ ] Move handle_vertexai_parallel_tools to google.py
- [ ] Move handle_genai_tools to google.py
- [ ] Move handle_genai_structured_outputs to google.py
- [ ] Move handle_cohere_tools to cohere.py
- [ ] Move handle_cohere_json_schema to cohere.py
- [ ] Move handle_cohere_modes to cohere.py
- [ ] Move handle_mistral_tools to mistral.py
- [ ] Move handle_mistral_structured_outputs to mistral.py
- [ ] Move handle_bedrock_json to bedrock.py
- [ ] Move handle_bedrock_tools to bedrock.py
- [ ] Move _prepare_bedrock_converse_kwargs_internal to bedrock.py
- [ ] Move handle_fireworks_tools to fireworks.py
- [ ] Move handle_fireworks_json to fireworks.py
- [ ] Move handle_cerebras_tools to cerebras.py
- [ ] Move handle_cerebras_json to cerebras.py
- [ ] Move handle_writer_tools to writer.py
- [ ] Move handle_writer_json to writer.py
- [ ] Move handle_perplexity_json to perplexity.py
- [ ] Move handle_tools to openai.py
- [ ] Move handle_tools_strict to openai.py
- [ ] Move handle_functions to openai.py
- [ ] Move handle_json_modes to openai.py
- [ ] Move handle_json_o1 to openai.py
- [ ] Move handle_parallel_tools to openai.py
- [ ] Move handle_responses_tools to openai.py
- [ ] Move handle_responses_tools_with_inbuilt_tools to openai.py
- [ ] Move handle_openrouter_structured_outputs to openai.py

## Phase 3: Update Imports and References
- [x] Update reask.py imports
- [x] Update process_response.py imports
- [x] Update utils.py for backwards compatibility
- [x] Update utils/__init__.py to export all functions
- [ ] Update retry.py imports
- [ ] Update all client_*.py files imports
- [ ] Update multimodal.py imports
- [ ] Update any other files that import from utils.py

## Phase 4: Create Handler Registries
- [x] Create ANTHROPIC_HANDLERS registry in anthropic.py
- [x] Create OPENAI_HANDLERS registry in openai.py
- [x] Create GOOGLE_HANDLERS registry in google.py
- [x] Create COHERE_HANDLERS registry in cohere.py
- [x] Create MISTRAL_HANDLERS registry in mistral.py
- [x] Create BEDROCK_HANDLERS registry in bedrock.py
- [x] Create FIREWORKS_HANDLERS registry in fireworks.py
- [x] Create CEREBRAS_HANDLERS registry in cerebras.py
- [x] Create WRITER_HANDLERS registry in writer.py
- [x] Create PERPLEXITY_HANDLERS registry in perplexity.py

## Phase 5: Testing and Cleanup
- [x] Run tests to verify refactoring
- [x] Fix circular import issues
- [ ] Remove deprecated code from original files
- [ ] Update documentation