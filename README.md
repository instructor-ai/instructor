# Instructor: Structured LLM Outputs for Python

Turn any LLM into a structured data extraction tool with a single line of code.

[![PyPI - Version](https://img.shields.io/pypi/v/instructor?style=flat-square&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square&logo=pypi&logoColor=white&label=Downloads)](https://pypi.org/project/instructor/)
[![GitHub Repo stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square&logo=github&logoColor=white)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square&logo=discord&logoColor=white&label=Discord)](https://discord.gg/bD9YE9JArw)
[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/jxnlco)

## Get structured data from any LLM

```python
import instructor
from pydantic import BaseModel

# Works with any provider: OpenAI, Anthropic, Google, Ollama, etc.
client = instructor.from_provider("openai/gpt-4o-mini")

class User(BaseModel):
    name: str
    age: int
    email: str

# That's it! Instructor handles validation, retries, and type safety
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Jason is 25, email jason@example.com"}]
)

print(user)  # User(name='Jason', age=25, email='jason@example.com')
```

## Why developers love Instructor

⚡ **Simple**: One API for 15+ LLM providers  
🛡️ **Reliable**: Automatic validation and retries with Pydantic  
🔄 **Flexible**: Streaming, partial outputs, and custom validators  
🎯 **Type-safe**: Full IDE support with autocomplete  

## Installation

Using uv (recommended):
```bash
uv add instructor
```

Using pip:
```bash
pip install instructor
```

## Quick Examples

### Extract structured data from any text

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool

product = client.chat.completions.create(
    response_model=ProductInfo,
    messages=[{"role": "user", "content": "iPhone 15 Pro Max 256GB - $999 (Available)"}]
)
print(product)  # ProductInfo(name='iPhone 15 Pro Max 256GB', price=999.0, in_stock=True)
```

### Streaming complex objects

```python
from typing import List
import instructor

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]

# Stream partial objects as they're generated
for partial_recipe in client.chat.completions.create(
    response_model=instructor.Partial[Recipe],
    messages=[{"role": "user", "content": "Write a recipe for chocolate chip cookies"}],
    stream=True
):
    print(partial_recipe)  # Prints gradually building recipe
```

### Automatic validation with retries

```python
from pydantic import BaseModel, field_validator

class Email(BaseModel):
    address: str
    
    @field_validator('address')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v

# Instructor automatically retries when validation fails
email = client.chat.completions.create(
    response_model=Email,
    messages=[{"role": "user", "content": "My email is jason at example dot com"}],
    max_retries=3  # Automatically fixes common LLM mistakes
)
print(email)  # Email(address='jason@example.com')
```

## Works with all major providers

Instructor supports 15+ LLM providers with a unified API:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic  
client = instructor.from_provider("anthropic/claude-3-5-sonnet-20241022")

# Google
client = instructor.from_provider("google/gemini-pro") 

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# Works exactly the same with all providers!
```

## Documentation

📚 [Comprehensive docs](https://python.useinstructor.com) with examples, tutorials, and best practices

### Get Started
- [Installation](https://python.useinstructor.com/#quick-start)
- [Core Concepts](https://python.useinstructor.com/concepts/models/)
- [Provider Integration](https://python.useinstructor.com/integrations/)

### Learn 
- [Why use Instructor?](https://python.useinstructor.com/why/)
- [Cookbook Examples](https://python.useinstructor.com/examples/)
- [Blog](https://python.useinstructor.com/blog/)

### Advanced Features
- [Validation & Retries](https://python.useinstructor.com/concepts/retrying/)
- [Streaming](https://python.useinstructor.com/concepts/partial/)
- [Async Support](https://python.useinstructor.com/concepts/parallel/)

Run `instructor docs` to open documentation in your browser.

## Available in Multiple Languages

Instructor's simple API is available across many programming languages:

- [Python](https://python.useinstructor.com) - The original implementation
- [TypeScript](https://js.useinstructor.com) - Full-featured JS/TS port  
- [Ruby](https://ruby.useinstructor.com) - Ruby implementation
- [Go](https://go.useinstructor.com) - Go implementation  
- [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
- [Rust](https://rust.useinstructor.com) - Rust implementation

All implementations follow the same simple patterns, making it easy to use Instructor in your language of choice.

## Contributing

We love contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started. Whether it's adding new features, improving documentation, or creating examples, we'd love your help.

See [CONTRIBUTING.md](https://github.com/instructor-ai/instructor/blob/main/CONTRIBUTING.md) for setup instructions.

## Support & Community

- 💬 [Join our Discord](https://discord.gg/bD9YE9JArw) for discussions and support
- 🐦 [Follow on Twitter](https://twitter.com/jxnlco) for updates
- ⭐ [Star on GitHub](https://github.com/instructor-ai/instructor) to support the project
- 🎯 [Browse examples](https://python.useinstructor.com/examples/) for inspiration

## Want your logo here?

If your company uses Instructor, we'd love to feature your logo! [Fill out this form](https://q7gjsgfstrp.typeform.com/to/wluQlVVQ).

## License

MIT License - See [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

## Citation

If you use Instructor in your research, please cite:

```bibtex
@software{liu2024instructor,
  author = {Jason Liu and Contributors},
  title = {Instructor: Structured outputs for LLMs},
  url = {https://github.com/instructor-ai/instructor},
  year = {2024}
}
```