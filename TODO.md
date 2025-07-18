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
- [x] Move extract_json_from_codeblock
- [x] Move extract_json_from_stream
- [x] Move extract_json_from_stream_async
- [x] Move update_total_usage
- [x] Move dump_message
- [x] Move is_async
- [x] Move merge_consecutive_messages
- [x] Move classproperty
- [x] Move get_message_content
- [x] Move disable_pydantic_error_url
- [x] Move is_typed_dict
- [x] Move prepare_response_model

### From utils.py to providers.py:
- [x] Move Provider enum
- [x] Move get_provider function

### From utils.py to anthropic.py:
- [x] Move SystemMessage TypedDict
- [x] Move combine_system_messages
- [x] Move extract_system_messages

### From utils.py to google.py:
- [x] Move transform_to_gemini_prompt
- [x] Move verify_no_unions
- [x] Move map_to_gemini_function_schema
- [x] Move update_genai_kwargs
- [x] Move update_gemini_kwargs
- [x] Move extract_genai_system_message
- [x] Move convert_to_genai_messages

### From reask.py to provider modules:
- [x] Move reask_anthropic_tools to anthropic.py
- [x] Move reask_anthropic_json to anthropic.py
- [x] Move reask_gemini_tools to google.py
- [x] Move reask_gemini_json to google.py
- [x] Move reask_vertexai_tools to google.py
- [x] Move reask_vertexai_json to google.py
- [x] Move reask_genai_tools to google.py
- [x] Move reask_genai_structured_outputs to google.py
- [x] Move reask_cohere_tools to cohere.py
- [x] Move reask_mistral_tools to mistral.py
- [x] Move reask_mistral_structured_outputs to mistral.py
- [x] Move reask_bedrock_json to bedrock.py
- [x] Move reask_fireworks_tools to fireworks.py
- [x] Move reask_fireworks_json to fireworks.py
- [x] Move reask_cerebras_tools to cerebras.py
- [x] Move reask_writer_tools to writer.py
- [x] Move reask_writer_json to writer.py
- [x] Move reask_perplexity_json to perplexity.py
- [x] Move reask_tools to openai.py
- [x] Move reask_responses_tools to openai.py
- [x] Move reask_md_json to openai.py
- [x] Move reask_default to openai.py

### From process_response.py to provider modules:
- [x] Move handle_anthropic_tools to anthropic.py
- [x] Move handle_anthropic_json to anthropic.py
- [x] Move handle_anthropic_reasoning_tools to anthropic.py
- [x] Move handle_anthropic_parallel_tools to anthropic.py
- [x] Move handle_gemini_tools to google.py
- [x] Move handle_gemini_json to google.py
- [x] Move handle_vertexai_tools to google.py
- [x] Move handle_vertexai_json to google.py
- [x] Move handle_vertexai_parallel_tools to google.py
- [x] Move handle_genai_tools to google.py
- [x] Move handle_genai_structured_outputs to google.py
- [x] Move handle_cohere_tools to cohere.py
- [x] Move handle_cohere_json_schema to cohere.py
- [x] Move handle_cohere_modes to cohere.py
- [x] Move handle_mistral_tools to mistral.py
- [x] Move handle_mistral_structured_outputs to mistral.py
- [x] Move handle_bedrock_json to bedrock.py
- [x] Move handle_bedrock_tools to bedrock.py
- [x] Move _prepare_bedrock_converse_kwargs_internal to bedrock.py
- [x] Move handle_fireworks_tools to fireworks.py
- [x] Move handle_fireworks_json to fireworks.py
- [x] Move handle_cerebras_tools to cerebras.py
- [x] Move handle_cerebras_json to cerebras.py
- [x] Move handle_writer_tools to writer.py
- [x] Move handle_writer_json to writer.py
- [x] Move handle_perplexity_json to perplexity.py
- [x] Move handle_tools to openai.py
- [x] Move handle_tools_strict to openai.py
- [x] Move handle_functions to openai.py
- [x] Move handle_json_modes to openai.py
- [x] Move handle_json_o1 to openai.py
- [x] Move handle_parallel_tools to openai.py
- [x] Move handle_responses_tools to openai.py
- [x] Move handle_responses_tools_with_inbuilt_tools to openai.py
- [x] Move handle_openrouter_structured_outputs to openai.py

## Phase 3: Update Imports and References
- [x] Update reask.py imports
- [x] Update process_response.py imports
- [x] Update utils.py for backwards compatibility
- [x] Update utils/__init__.py to export all functions
- [x] Verify all imports are working correctly

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
- [x] Remove duplicate handler functions from process_response.py
- [x] Import handlers from provider modules in process_response.py
- [x] Final test run to ensure everything works
- [x] Auto-fix unused imports with ruff
- [ ] Update documentation (if needed)