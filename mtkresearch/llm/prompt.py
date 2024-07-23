import json
import string
import random


class MRPromptV1:
    def __init__(self, bos_token='<s>', eos_token='</s>'):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.instruct_tokens = ['[INST]', '[/INST]']
        self.func_tokens = ['[FUNC]', '[/FUNC]']
        self.call_tokens = ['[FUNC_CALL]', '[/FUNC_CALL]']
        self.result_tokens = ['[FUNC_RESULT]', '[/FUNC_RESULT]']

    def _font(self, sys=None):
        if sys is None or not sys.strip():
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        return f'{self.bos_token} {sys} '

    def _font_with_functions(self, sys, functions):
        if sys is None:
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        functions = json.dumps(functions, ensure_ascii=False)
        return f'{self.bos_token} {self.func_tokens[0]} {functions} {self.func_tokens[1]} {sys} '

    def check_conversations(self, conversations, functions=None):
        if functions is not None:
            function_names = [func['name'] for func in functions]
        for i, conv in enumerate(conversations):
            role = conv['role']
            if role == 'system':
                if i != 0:
                    raise ValueError
                if not isinstance(conv['content'], str):
                    raise ValueError
                if conversations[1]['role'] != 'user':
                    raise ValueError

            elif role == 'user':
                if not isinstance(conv['content'], str):
                    raise ValueError
                if i != 0:
                    if conversations[i - 1]['role'] == 'user':
                        raise ValueError
                    if conversations[i - 1]['role'] == 'assistant' and 'tool_calls' in conversations[i - 1]:
                        raise ValueError

            elif role == 'assistant' and 'tool_calls' not in conv:
                if i == 0:
                    raise ValueError
                elif not(conversations[i - 1]['role'] == 'user' or conversations[i - 1]['role'] == 'tool'):
                    raise ValueError

                if not isinstance(conv['content'], str):
                    raise ValueError

            elif role == 'assistant' and 'tool_calls' in conv:
                if i == 0:
                    raise ValueError
                elif not(conversations[i - 1]['role'] == 'user' or conversations[i - 1]['role'] == 'tool'):
                    raise ValueError

                if not functions:
                    raise ValueError

                for tool_call in conv['tool_calls']:
                    if tool_call['type'] != 'function':
                        raise ValueError
                    json.loads(tool_call['function']['arguments'])
                    if tool_call['function']['name'] not in function_names:
                        raise ValueError

            elif role == 'tool':
                if i == 0:
                    raise ValueError
                elif not ((conversations[i - 1]['role'] == 'assistant' and 'tool_calls' in conversations[i - 1]) or (conversations[i - 1]['role'] == 'tool')):
                    raise ValueError

                if not functions:
                    raise ValueError

                json.loads(conv['content'])

                tool_call_id = conv['tool_call_id']
                name = conv['name']
                # go to corresponding calls
                j = i - 1
                while j >= 0:
                    if conversations[j]['role'] == 'assistant' and 'tool_calls' in conversations[j]:
                        break
                    elif conversations[j]['role'] != 'tool':
                        raise ValueError
                    j -= 1
                if j < 0:
                    raise ValueError
                corresponding_tool_calls = conversations[j]['tool_calls']
                corresponding_ids = [c['id'] for c in corresponding_tool_calls]
                k = corresponding_ids.index(tool_call_id)
                if k < 0:
                    raise ValueError
                if corresponding_tool_calls[k]['function']['name'] != name:
                    raise ValueError
                
    def check_functions(self, functions):
        for func in functions:
            if 'name' not in func or 'description' not in func or 'parameters' not in func:
                raise ValueError
            if not isinstance(func['name'], str) or not isinstance(func['description'], str):
                raise ValueError
            if not (func['parameters'] is None or isinstance(func['parameters'], dict)):
                raise ValueError
            if func['parameters'] is None or len(func['parameters']) == 0:
                continue
            if 'type' not in func['parameters'] or 'properties' not in func['parameters']:
                raise ValueError
            if not isinstance(func['parameters']['properties'], dict):
                raise ValueError
            if 'required' in func['parameters']:
                if not isinstance(func['parameters']['required'], list):
                    raise ValueError
                for name in func['parameters']['required']:
                    if name not in func['parameters']['properties']:
                        raise ValueError

    def get_prompt(self, conversations, functions=None):
        
        if functions:
            self.check_functions(functions)
            self.check_conversations(conversations, functions=functions)
        else:
            self.check_conversations(conversations)

        prompt = ''
        sys = None
        if conversations[0]['role'] == 'system':
            sys = conversations[0]['content']
            conversations = conversations[1:]

        if functions:
            prompt += self._font_with_functions(sys, functions)
        else:
            prompt += self._font(sys)

        func_agg = []
        for i, conv in enumerate(conversations):
            if conv['role'] == 'user':
                prompt += f'{self.instruct_tokens[0]} {conv["content"].strip()} {self.instruct_tokens[1]} '
            elif conv['role'] == 'assistant' and 'tool_calls' not in conv:
                prompt += conv['content'].strip() + self.eos_token
            elif conv['role'] == 'assistant' and 'tool_calls' in conv:
                tool_calls = conv['tool_calls']

                if i + 1 == len(conversations):
                    tool_calls_str = '[' \
                        + ', '.join(['{' + f'"name": "{c["function"]["name"]}", "arguments": {json.dumps(json.loads(c["function"]["arguments"]))}' + '}' for c in tool_calls]) \
                        + ']'
                else:
                    tool_calls_str = '[' \
                        + ', '.join(['{' + f'"call_id": "{c["id"]}", "name": "{c["function"]["name"]}", "arguments": {json.dumps(json.loads(c["function"]["arguments"]))}' + '}' for c in tool_calls]) \
                        + ']'
                prompt += f'{self.call_tokens[0]} {tool_calls_str} {self.call_tokens[1]}'

            elif conv['role'] == 'tool':
                func_agg.append(
                    '{"call_id": "' + conv['tool_call_id'] + '", "name": "' + conv['name'] + '", "content": ' + json.dumps(json.loads(conv['content'])) + '}'
                )
                if i + 1 == len(conversations) or conversations[i + 1]['role'] != 'tool':
                    prompt = prompt.rstrip(self.call_tokens[1])
                    prompt += f'{self.result_tokens[0]} [' + ', '.join(func_agg) + f'] {self.result_tokens[1]} '

        return prompt
    
    def generate_call_id(self):
        length = 24
        pool = string.ascii_letters + string.digits
        key = ''.join(random.choice(pool) for i in range(length))
        return f'call_{key}'

    def parse_generated_str(self, generated_str):
        generated_str = generated_str.strip()
        if generated_str.startswith(self.call_tokens[0]): # function call
            text = generated_str[len(self.call_tokens[0]):].lstrip()
            if self.call_tokens[1] in text:
                text = text.split(self.call_tokens[1])[0].rstrip()
            func_calls = eval(text)
            for i in range(len(func_calls)):
                func_calls[i]['arguments'] = json.dumps(func_calls[i]['arguments'])
            conv = {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'id': self.generate_call_id(),
                        'type': 'function',
                        'function': func_call
                    } for func_call in func_calls]
            }
        else:
            conv = {
                'role': 'assistant',
                'content': generated_str.rstrip(self.eos_token)
            }
        return conv



class MRPromptV2(MRPromptV1):
    def __init__(self, bos_token='<s>', eos_token='</s>',
                 instance_start_token='<|im_start|>', instance_end_token='<|im_end|>',
                 tool_call_begin_token='<|tool_call_begin|>', tool_call_end_token='<|tool_call_end|>'):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.instance_start_token = instance_start_token
        self.instance_end_token = instance_end_token
        self.tool_call_begin_token = tool_call_begin_token
        self.tool_call_end_token = tool_call_end_token

        self.system_role = 'system'
        self.user_role = 'user'
        self.assistant_role = 'assistant'
        self.tools_role = 'tools'
        self.tool_response_role = 'tool_response'

    def _font(self, sys=None):
        if sys is None or not sys.strip():
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        return f'{self.bos_token}{self.instance_start_token}{self.system_role}\n{sys}{self.instance_end_token}'

    def _font_with_functions(self, sys, functions):
        if sys is None:
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        functions = json.dumps(functions, ensure_ascii=False)
        return f'{self.bos_token}{self.instance_start_token}{self.tools_role}\n{functions}{self.instance_end_token}' + \
            f'{self.instance_start_token}{self.system_role}\n{sys}{self.instance_end_token}'

    def get_prompt(self, conversations, functions=None):
        
        if functions:
            self.check_functions(functions)
            self.check_conversations(conversations, functions=functions)
        else:
            self.check_conversations(conversations)

        prompt = ''
        sys = None
        if conversations[0]['role'] == 'system':
            sys = conversations[0]['content']
            conversations = conversations[1:]

        if functions:
            prompt += self._font_with_functions(sys, functions)
        else:
            prompt += self._font(sys)

        for i, conv in enumerate(conversations):
            if conv['role'] == 'user':
                prompt += f'{self.instance_start_token}{self.user_role}\n{conv["content"].strip()}{self.instance_end_token}' + \
                    f'{self.instance_start_token}{self.assistant_role}\n'

            elif conv['role'] == 'assistant' and 'tool_calls' not in conv:
                prompt += conv['content'].strip() + self.instance_end_token

            elif conv['role'] == 'assistant' and 'tool_calls' in conv:
                tool_calls = conv['tool_calls']

                if i + 1 == len(conversations):
                    tool_calls_str = f'{self.tool_call_end_token}{self.tool_call_begin_token}'.join([
                        json.dumps({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}) for c in tool_calls
                    ])
                else:
                    tool_calls_str = f'{self.tool_call_end_token}{self.tool_call_begin_token}'.join([
                        json.dumps({"call_id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}) for c in tool_calls
                    ])

                prompt += f'{self.tool_call_begin_token}{tool_calls_str}{self.tool_call_end_token}{self.instance_end_token}'

            elif conv['role'] == 'tool':
                tool_response_str = json.dumps({"call_id": conv['tool_call_id'], "name": conv['name'], "content": conv['content']})
                prompt += f'{self.instance_start_token}{self.tool_response_role}\n{tool_response_str}{self.instance_end_token}'

                if i + 1 == len(conversations) or conversations[i + 1]['role'] != 'tool':
                    prompt += f'{self.instance_start_token}{self.assistant_role}\n'

        return prompt

    def parse_generated_str(self, generated_str):
        generated_str = generated_str.strip()
        if generated_str.endswith(self.instance_end_token):
            generated_str = generated_str[:-len(self.instance_end_token)]

        if self.tool_call_begin_token in generated_str: # function call
            tool_calls = []

            for segment in generated_str.split(self.tool_call_begin_token)[1:]:
                if not segment.endswith(self.tool_call_end_token):
                    raise ValueError
                func_call = json.loads(segment[:-len(self.tool_call_end_token)])
                func_call['arguments'] = func_call['arguments']
                tool_calls.append({
                    'id': self.generate_call_id(),
                    'type': 'function',
                    'function': func_call
                })
            conv = {
                'role': 'assistant',
                'tool_calls': tool_calls
            }
        else:
            conv = {
                'role': 'assistant',
                'content': generated_str
            }
        return conv
