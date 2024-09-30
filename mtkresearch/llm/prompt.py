import json
import string
import random
import sys


def _removeprefix(content, prefix):
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
        return content.removeprefix(prefix)
    else:
        return content[len(prefix):] if content.startswith(prefix) else content


def _removesuffix(content, suffix):
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
        return content.removesuffix(suffix)
    else:
        return content[:-len(suffix)] if content.endswith(suffix) else content


class MRPromptV1:
    def __init__(self, bos_token='<s>', eos_token='</s>'):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.instruct_tokens = ['[INST]', '[/INST]']
        self.func_tokens = ['[FUNC]', '[/FUNC]']
        self.call_tokens = ['[FUNC_CALL]', '[/FUNC_CALL]']
        self.result_tokens = ['[FUNC_RESULT]', '[/FUNC_RESULT]']

    def _font(self, sys=None, add_bos_token=False):
        if sys is None or not sys.strip():
            sys = 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.'
        sys = sys.strip()
        return f'{self.bos_token}{sys} ' if add_bos_token else f'{sys} '

    def _check_arguments(self, arguments, func_description):
        errors = []
        param_details = func_description['parameters']['properties']
        required_params = func_description['parameters'].get('required', [])
        for param in required_params:
            if param not in arguments:
                errors.append(f"Missing required parameter: '{param}'")
        
        for param, value in arguments.items():
            if param not in param_details:
                errors.append(f"Unexpected parameter: '{param}'")
                continue
            expected_type = param_details[param]['type']

            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"Incorrect type for '{param}': Expected string, got {type(value).__name__}")
            elif expected_type == 'integer' and not isinstance(value, int):
                errors.append(f"Incorrect type for '{param}': Expected integer, got {type(value).__name__}")
            elif expected_type == 'float' and not isinstance(value, float):
                errors.append(f"Incorrect type for '{param}': Expected float, got {type(value).__name__}")
            elif expected_type == 'boolean' and not isinstance(value, bool):
                errors.append(f"Incorrect type for '{param}': Expected boolean, got {type(value).__name__}")
            elif expected_type == 'array' and not isinstance(value, list):
                errors.append(f"Incorrect type for '{param}': Expected array, got {type(value).__name__}")

            if 'enum' in param_details[param]:
                if value not in param_details[param]['enum']:
                    errors.append(f"Incorrect value for '{param}': Expected one of {param_details[param]['enum']}, got '{value}'")

        if errors:
            raise ValueError('\n'.join(errors))

    def check_conversations(self, conversations, functions=None):
        if functions is not None:
            function_mapping = {func['name']: func for func in functions}

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

            elif role == 'assistant' and 'tool_calls' not in conv: # assistant answer
                if i == 0:
                    raise ValueError
                elif not(conversations[i - 1]['role'] == 'user' or conversations[i - 1]['role'] == 'tool'):
                    raise ValueError

                if not isinstance(conv['content'], str):
                    raise ValueError

            elif role == 'assistant' and 'tool_calls' in conv: # assistant tool call
                if i == 0:
                    raise ValueError
                elif not(conversations[i - 1]['role'] == 'user' or conversations[i - 1]['role'] == 'tool'):
                    raise ValueError

                if not functions:
                    raise ValueError

                for tool_call in conv['tool_calls']:
                    if tool_call['type'] != 'function':
                        raise ValueError
                    arguments = json.loads(tool_call['function']['arguments'])
                    name = tool_call['function']['name']
                    if name not in function_mapping:
                        raise ValueError
                    self._check_arguments(arguments, function_mapping[name])

            elif role == 'tool': # tool response
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

    def get_prompt(self, conversations, add_bos_token=False):
        self.check_conversations(conversations)

        prompt = ''
        sys = None
        if conversations[0]['role'] == 'system':
            sys = conversations[0]['content']
            conversations = conversations[1:]

        prompt += self._font(sys, add_bos_token)

        for i, conv in enumerate(conversations):
            if conv['role'] == 'user':
                prompt += f' {self.instruct_tokens[0]} {conv["content"].strip()} {self.instruct_tokens[1]} '
            elif conv['role'] == 'assistant' and 'tool_calls' not in conv:
                prompt += conv['content'].strip()
                if i == len(conversations) - 1:
                    prompt += self.eos_token

        return prompt

    def parse_generated_str(self, generated_str):
        generated_str = generated_str.strip()
        conv = {
            'role': 'assistant',
            'content': _removesuffix(generated_str, self.eos_token)
        }
        return conv


class MRPromptV2(MRPromptV1):
    def __init__(self, bos_token='<s>', eos_token='</s>',
                 instance_start_token='<|im_start|>', instance_end_token='<|im_end|>',
                 tool_call_token='<|use_tool|>', answer_token='<|answer|>',
                 tool_call_begin_token='<|tool_call_begin|>', tool_call_end_token='<|tool_call_end|>'):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.instance_start_token = instance_start_token
        self.instance_end_token = instance_end_token
        self.tool_call_token = tool_call_token
        self.answer_token = answer_token
        self.tool_call_begin_token = tool_call_begin_token
        self.tool_call_end_token = tool_call_end_token

        self.system_role = 'system'
        self.user_role = 'user'
        self.assistant_role = 'assistant'
        self.tools_role = 'tools'
        self.tool_response_role = 'tool_response'

    def _font(self, sys=None, add_bos_token=False):
        if sys is None or not sys.strip():
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        prompt = f'{self.instance_start_token}{self.system_role}\n{sys}{self.instance_end_token}'
        return self.bos_token + prompt if add_bos_token else prompt

    def _font_with_functions(self, sys, functions, add_bos_token=False):
        if sys is None:
            sys = 'You are a helpful assistant.'
        sys = sys.strip()
        functions = json.dumps(functions, ensure_ascii=False)
        prompt = f'{self.instance_start_token}{self.tools_role}\n{functions}{self.instance_end_token}' + \
            f'{self.instance_start_token}{self.system_role}\n{sys}{self.instance_end_token}'
        return self.bos_token + prompt if add_bos_token else prompt
    
    def generate_call_id(self):
        length = 24
        pool = string.ascii_letters + string.digits
        key = ''.join(random.choice(pool) for i in range(length))
        return f'call_{key}'

    def get_prompt(self, conversations, functions=None, add_bos_token=False):
        config = {
            'add_decision_token': True,
            'add_reason': False,
        }
        
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
            prompt += self._font_with_functions(sys, functions, add_bos_token=add_bos_token)
        else:
            prompt += self._font(sys, add_bos_token=add_bos_token)

        for i, conv in enumerate(conversations):
            if conv['role'] == 'user':
                prompt += f'{self.instance_start_token}{self.user_role}\n{conv["content"].strip()}{self.instance_end_token}' + \
                    f'{self.instance_start_token}{self.assistant_role}\n'

            elif conv['role'] == 'assistant' and 'tool_calls' not in conv:
                appended_prompt = conv['content'].strip() + self.instance_end_token
                if config['add_decision_token']:
                    appended_prompt = self.answer_token + appended_prompt
                prompt += appended_prompt

            elif conv['role'] == 'assistant' and 'tool_calls' in conv:
                tool_calls = conv['tool_calls']

                if i + 1 == len(conversations):
                    tool_calls_str = f'{self.tool_call_end_token}{self.tool_call_begin_token}'.join([
                        json.dumps({
                            "name": c["function"]["name"],
                            "arguments": json.dumps(json.loads(c["function"]["arguments"]), ensure_ascii=False)
                        }, ensure_ascii=False)
                        for c in tool_calls
                    ])
                else:
                    tool_calls_str = f'{self.tool_call_end_token}{self.tool_call_begin_token}'.join([
                        json.dumps({
                            "call_id": c["id"], 
                            "name": c["function"]["name"],
                            "arguments": json.dumps(json.loads(c["function"]["arguments"]), ensure_ascii=False)
                        }, ensure_ascii=False)
                        for c in tool_calls
                    ])

                appended_prompt = f'{self.tool_call_begin_token}{tool_calls_str}{self.tool_call_end_token}{self.instance_end_token}'
                
                if config['add_reason']:
                    appended_prompt = conv.get('reason', '') + appended_prompt
                if config['add_decision_token']:
                    appended_prompt = self.tool_call_token + appended_prompt
                prompt += appended_prompt

            elif conv['role'] == 'tool':
                tool_response_str = json.dumps(
                    {
                        "call_id": conv['tool_call_id'],
                        "name": conv['name'],
                        "content": json.dumps(json.loads(conv['content']), ensure_ascii=False)
                    }, ensure_ascii=False)
                prompt += f'{self.instance_start_token}{self.tool_response_role}\n{tool_response_str}{self.instance_end_token}'

                if i + 1 == len(conversations) or conversations[i + 1]['role'] != 'tool':
                    prompt += f'{self.instance_start_token}{self.assistant_role}\n'

        return prompt

    def parse_generated_str(self, generated_str):
        generated_str = generated_str.strip()
        generated_str = _removeprefix(generated_str, self.answer_token).strip()
        generated_str = _removeprefix(generated_str, self.tool_call_token).strip()
        generated_str = _removesuffix(generated_str, self.instance_end_token).strip()

        if self.tool_call_begin_token in generated_str: # function call
            try:
                tool_calls = []

                for segment in generated_str.split(self.tool_call_begin_token)[1:]:
                    if not segment.endswith(self.tool_call_end_token):
                        raise ValueError

                    func_call = json.loads(_removesuffix(segment, self.tool_call_end_token).strip())
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
            except Exception as e:
                print(f'skip error: {e}')
                conv = {
                    'role': 'assistant',
                    'content': ''
                }
        else:
            conv = {
                'role': 'assistant',
                'content': generated_str
            }
        return conv
