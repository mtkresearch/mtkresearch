# MTK Research Package

# Setup

```
$ pip install .
```

# Prompt for chat

```python
from mtkresearch.llm.prompt import MRPrompt

prompt = MRPrompt()

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
from mtkresearch.llm.prompt import MRPrompt

prompt = MRPrompt()

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

# Demo on VLLM

```python
from vllm import LLM, SamplingParams
from mtkresearch.llm.chat import MRChatManager
from mtkresearch.llm.prompt import MRPromptV2

functions = [
  {'name': 'generate_random_password',
  'description': 'Generate a random password with specified length',
  'parameters': {'type': 'object',
   'properties': {'length': {'type': 'integer',
     'description': 'The desired length of the password'}},
   'required': ['length']}},
 {'name': 'search_recipe',
  'description': 'Search for a recipe based on ingredients',
  'parameters': {'type': 'object',
   'properties': {'ingredients': {'type': 'array',
     'items': {'type': 'string'},
     'description': 'List of ingredients to search for'}},
   'required': ['ingredients']}},
 {'name': 'search_jobs',
  'description': 'Search for job opportunities based on keywords',
  'parameters': {'type': 'object',
   'properties': {'keywords': {'type': 'array',
     'items': {'type': 'string'},
     'description': 'The keywords to search for in job titles or descriptions'},
    'location': {'type': 'string',
     'description': 'The location to filter the job search results'}},
   'required': ['keywords']}}
]

llm = LLM(model="/path/to/model")
params = SamplingParams(temperature=0.1, top_p=0.1, max_tokens=512, stop_token_ids=[llm.get_tokenizer().convert_tokens_to_ids('<|im_end|>')])
pp = MRPromptV2(bos_token='<s>', eos_token='</s>')

def _run(ma, pp, params):
    output_text = llm.generate(pp.get_prompt(ma.conversations, ma.functions).removeprefix('<s>'), 
        params)[0].outputs[0].text
    print('response:', ma.parse_assistant(output_text))

ma = MRChatManager(prompt=pp, functions=functions)

ma.user_input('Please give me the jobs about AI research')
_run(ma, pp, params)

ma.func_response('call_TP5DOBx3IlnyLsaijrvvYxTf', {"jobs": ["MTK Research / AI Researcher"]})
_run(ma, pp, params)
```


