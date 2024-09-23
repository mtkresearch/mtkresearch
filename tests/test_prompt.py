import json

from mtkresearch.llm.prompt import MRPromptV1, MRPromptV2


class TestMRPromptV1:
    prompt = MRPromptV1()
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

    def test_chat_case1(self): # sys + query
        conversations = [
            {
                "role": "system",
                "content": "SYS"
            },
            {
                "role": "user",
                "content": "QUERY1"
            }
        ]
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == '<s>SYS  [INST] QUERY1 [/INST] '

    def test_chat_case2(self): # sys + query + response
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
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == '<s>SYS  [INST] QUERY1 [/INST] RESPONSE1</s>'

    def test_chat_case3(self): # query + response + query
        conversations = [
            {
                "role": "user",
                "content": "QUERY1"
            },
            {
                "role": "assistant",
                "content": "RESPONSE1"
            },
            {
                "role": "user",
                "content": "QUERY2"
            }
        ]
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == '<s>You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.  [INST] QUERY1 [/INST] RESPONSE1 [INST] QUERY2 [/INST] ', result

    def test_parse_generated_str(self):
        generated_str = 'RESP1</s>'
        assert self.prompt.parse_generated_str(generated_str) == {
            'role': 'assistant',
            'content': 'RESP1'
        }


class TestMRPromptV2:
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

    def test_chat_case1(self): # sys + query
        conversations = [
            {
                "role": "system",
                "content": "SYS"
            },
            {
                "role": "user",
                "content": "QUERY1"
            }
        ]
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == (
            '<s><|im_start|>system\nSYS<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_chat_case2(self): # sys + query + response
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
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == (
            '<s><|im_start|>system\nSYS<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|answer|>RESPONSE1<|im_end|>'
        )

    def test_chat_case3(self): # query + response + query
        conversations = [
            {
                "role": "user",
                "content": "QUERY1"
            },
            {
                "role": "assistant",
                "content": "RESPONSE1"
            },
            {
                "role": "user",
                "content": "QUERY2"
            }
        ]
        result = self.prompt.get_prompt(conversations, add_bos_token=True)
        assert result == (
            '<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|answer|>RESPONSE1<|im_end|>'
            '<|im_start|>user\nQUERY2<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_func_case1(self): # sys + query + f_call

        conversations = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "QUERY1"
            },
            {
                "role": "assistant",
                "reason": "REASON",
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
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|><|im_end|>'
        )

    def test_func_case2(self): # sys + query + f_call + f_response
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
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nSYS<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|><|im_end|>'
            '<|im_start|>tool_response\n{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "content": "{\\"temperature\\": \\"22 celsius\\"}"}<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_func_case3(self): # sys + query + f_call + f_response + response + query
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
                "content": "RESP1"
            },
            {
                "role": "user",
                "content": "QUERY2"
            },
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nSYS<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|><|im_end|>'
            '<|im_start|>tool_response\n{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "content": "{\\"temperature\\": \\"22 celsius\\"}"}<|im_end|>'
            '<|im_start|>assistant\n<|answer|>RESP1<|im_end|>'
            '<|im_start|>user\nQUERY2<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_func_case4(self): # sys + query + response + query + f_call
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
                "content": "RESP1"
            },
            {
                "role": "user",
                "content": "QUERY2"
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
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nSYS<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|answer|>RESP1<|im_end|>'
            '<|im_start|>user\nQUERY2<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|><|im_end|>'
        )

    def test_func_case5(self): # query + f_call(2)
        conversations = [
            {
                "role": "user",
                "content": "QUERY1"
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        'type': 'function',
                        'function': {
                            'arguments': "{\"location\": \"Boston, MA\"}",
                            'name': 'get_current_weather'
                        }
                    },
                    {
                        'type': 'function',
                        'function': {
                            'arguments': "{\"location\": \"Taipei, Taiwan\"}",
                            'name': 'get_current_weather'
                        }
                    }]
            },
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|>'
            '<|tool_call_begin|>{"name": "get_current_weather", "arguments": "{\\"location\\": \\"Taipei, Taiwan\\"}"}<|tool_call_end|><|im_end|>'
        )

    def test_func_case6(self): # query + f_call(2) + f_response(2)
        conversations = [
            {
                "role": "user",
                "content": "QUERY1"
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        'type': 'function',
                        'id': 'ID1',
                        'function': {
                            'arguments': "{\"location\": \"Boston, MA\"}",
                            'name': 'get_current_weather'
                        }
                    },
                    {
                        'type': 'function',
                        'id': 'ID2',
                        'function': {
                            'arguments': "{\"location\": \"Taipei, Taiwan\"}",
                            'name': 'get_current_weather'
                        }
                    }]
            },
            {
                "role": "tool",
                "tool_call_id": "ID1",
                "name": "get_current_weather",
                "content": "{\"temperature\": \"22 celsius\"}"
            },
            {
                "role": "tool",
                "tool_call_id": "ID2",
                "name": "get_current_weather",
                "content": "{\"temperature\": \"31 celsius\"}"
            },
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            '<|im_start|>user\nQUERY1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"call_id": "ID1", "name": "get_current_weather", "arguments": "{\\"location\\": \\"Boston, MA\\"}"}<|tool_call_end|>'
            '<|tool_call_begin|>{"call_id": "ID2", "name": "get_current_weather", "arguments": "{\\"location\\": \\"Taipei, Taiwan\\"}"}<|tool_call_end|><|im_end|>'
            '<|im_start|>tool_response\n{"call_id": "ID1", "name": "get_current_weather", "content": "{\\"temperature\\": \\"22 celsius\\"}"}<|im_end|>'
            '<|im_start|>tool_response\n{"call_id": "ID2", "name": "get_current_weather", "content": "{\\"temperature\\": \\"31 celsius\\"}"}<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_func_tc(self): # sys + query + f_call + f_response + response + query
        conversations = [
            {
                "role": "system",
                "content": "系統"
            },
            {
                "role": "user",
                "content": "問1"
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        'id': 'call_8jLWqlXaY3OisD24IHJLwD3G',
                        'type': 'function',
                        'function': {
                            'arguments': "{\"location\": \"波士頓\"}",
                            'name': 'get_current_weather'
                        }
                    }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_8jLWqlXaY3OisD24IHJLwD3G",
                "name": "get_current_weather",
                "content": "{\"temperature\": \"22 度c\"}"
            },
            {
                "role": "assistant",
                "content": "答1"
            },
            {
                "role": "user",
                "content": "問2"
            },
        ]
        result = self.prompt.get_prompt(conversations, self.functions, add_bos_token=True)
        assert result == (
            '<s><|im_start|>tools\n[{"name": "get_current_weather", "description": "Get the current_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", '
            '"description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]<|im_end|>'
            '<|im_start|>system\n系統<|im_end|>'
            '<|im_start|>user\n問1<|im_end|>'
            '<|im_start|>assistant\n<|use_tool|><|tool_call_begin|>{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "arguments": "{\\"location\\": \\"波士頓\\"}"}<|tool_call_end|><|im_end|>'
            '<|im_start|>tool_response\n{"call_id": "call_8jLWqlXaY3OisD24IHJLwD3G", "name": "get_current_weather", "content": "{\\"temperature\\": \\"22 度c\\"}"}<|im_end|>'
            '<|im_start|>assistant\n<|answer|>答1<|im_end|>'
            '<|im_start|>user\n問2<|im_end|>'
            '<|im_start|>assistant\n'
        )

    def test_parse_generated_str(self):
        generated_str = '<|answer|>RESP1<|im_end|>'
        assert self.prompt.parse_generated_str(generated_str) == {
            'role': 'assistant',
            'content': 'RESP1'
        }

        generated_str = '<|use_tool|><|tool_call_begin|>{"name": "get_ans", "arguments": "{\\"determine\\": true}"}<|tool_call_end|><|tool_call_begin|>{"name": "get_current_weather", "arguments": "{\\"location\\": \\"Taipei, Taiwan\\"}"}<|tool_call_end|><|im_end|>'
        conv = self.prompt.parse_generated_str(generated_str)
        
        print(conv)
        assert conv['role'] == 'assistant'
        assert conv['tool_calls'][0]['type'] == 'function'
        assert conv['tool_calls'][0]['function']['name'] == 'get_ans'
        assert conv['tool_calls'][0]['function']['arguments'] == '{"determine": true}'
        assert json.loads(conv['tool_calls'][0]['function']['arguments']) == {'determine': True}
        assert conv['tool_calls'][1]['type'] == 'function'
        assert conv['tool_calls'][1]['function']['name'] == 'get_current_weather'
        assert conv['tool_calls'][1]['function']['arguments'] == '{"location": "Taipei, Taiwan"}'
