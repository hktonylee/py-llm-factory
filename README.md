# Welcome

llm-factory is a python package that can select and install LLM provider dynamically.

# Usage

```python
import py_llm_factory

# By default, it will use the LLM provider from the environment variables.
client = py_llm_factory.get_openai_compatible_client()

# You can also specify the LLM provider and API key.
client = py_llm_factory.get_openai_compatible_client(provider='openai', api_key='your_api_key')

# Let's say if you want to use the Groq provider, you can do this:
client = py_llm_factory.get_openai_compatible_client(provider='groq', api_key='your_api_key')

# Or AWS Bedrock provider, you can do this:
client = py_llm_factory.get_openai_compatible_client(provider='openai', api_key='your_api_key', base_url='https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1')
```
