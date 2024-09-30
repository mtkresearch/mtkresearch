# MTK Research Package

# Setup

```
$ pip install mtkresearch
```

# Prompt for chat

```python
from mtkresearch.llm.prompt import MRPromptV2

prompt = MRPromptV2()

conversations = [
    {
        "role": "system",
        "content": "SYS"
    },
    {
        "role": "user",
        "content": "QUERY1"
    },
    {
        "role": "assistant",
        "content": "RESPONSE1"
    },
]

text = self.prompt.get_prompt(conversations)
```

# Prompt for function calling

```python
from mtkresearch.llm.prompt import MRPromptV2

prompt = MRPromptV2()

functions = [
    {
        'name': 'get_current_weather',
        'description': 'Get the current_weather',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                }
            },
            'required': ['location']
        }
    }
]
conversations = [
    {
        "role": "user",
        "content": "What's the weather in Boston?"
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                'id': 'call_8jLWqlXaY3OisD24IHJLwD3G',
                'type': 'function',
                'function': {
                    'arguments': "{\"location\": \"Boston, MA\"}",
                    'name': 'get_current_weather'
                }
            }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_8jLWqlXaY3OisD24IHJLwD3G",
        "name": "get_current_weather",
        "content": "{\"temperature\": \"22 celsius\"}"
    },
    {
        "role": "assistant",
        "content": "The temperature is 22 celsius in Boston."
    },
]

text = self.prompt.get_prompt(conversations, functions)
```
