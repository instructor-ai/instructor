# Client Streaming Support Matrix

| Client | Partial Streaming | Iterable Streaming | Notes |
|--------|------------------|-------------------|--------|
| Anthropic | ❌ | ❌ | 'AsyncAnthropic' object has no attribute 'chat' |
| Openai | ❌ | ❌ | The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable |
| Mistral | ❌ | ❌ | Mistral client not installed |

## Notes

- ✅ = Full support
- ❌ = Not supported or failed
