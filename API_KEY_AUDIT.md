# API Key Usage Audit

## Environment Variables Used

### Primary Provider API Keys
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic/Claude API access
- `GOOGLE_API_KEY` - Google/Gemini API access
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API access

### Secondary Provider API Keys
- `MISTRAL_API_KEY` - MistralAI API access
- `GROQ_API_KEY` - Groq API access
- `CEREBRAS_API_KEY` - Cerebras API access
- `FIREWORKS_API_KEY` - Fireworks API access
- `PERPLEXITY_API_KEY` - Perplexity API access
- `COHERE_API_KEY` - Cohere API access
- `WRITER_API_KEY` - Writer API access
- `XAI_API_KEY` - xAI API access
- `DEEPSEEK_API_KEY` - DeepSeek API access
- `TOGETHER_API_KEY` - Together API access
- `ANYSCALE_API_KEY` - Anyscale API access
- `RUNPOD_API_KEY` - RunPod API access
- `OPENROUTER_API_KEY` - OpenRouter API access

### Service API Keys
- `WATSONX_API_KEY` - IBM Watson API access
- `LANGCHAIN_API_KEY` - LangSmith API access
- `PAREA_API_KEY` - Parea API access

## Files with API Key Usage

### Core Files
- `instructor/auto_client.py` - Auto-detection of provider API keys
- `instructor/cli/usage.py` - OpenAI usage tracking

### Examples
- `examples/batch_api/run_batch_test.py` - Multiple provider API key checking
- `examples/batch/run_openai.py` - OpenAI API key validation
- `examples/batch/run_anthropic.py` - Anthropic API key usage
- Many other examples use various provider API keys

### Documentation
- Multiple docs files show API key setup instructions
- `docs/getting-started.md` - Setup instructions for multiple providers

## Security Considerations

✅ **Good Practices Found:**
- API keys read from environment variables (not hardcoded)
- API key validation before use
- Clear error messages when API keys are missing
- Redaction markers for sensitive data: `[REDACTED:api-key]`

⚠️ **Areas to Monitor:**
- Some examples pass API keys as parameters
- Documentation shows placeholder API key examples
- Test files may contain mock API keys

## Recommendations

1. **Continue using environment variables** for all API key access
2. **Validate API keys** before making API calls
3. **Use consistent naming** for API key environment variables
4. **Add API key validation** to auto_client.py for all providers
5. **Ensure no hardcoded API keys** in any files
