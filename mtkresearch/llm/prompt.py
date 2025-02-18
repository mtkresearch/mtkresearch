import json
import string
import random
import sys
import ast
from datetime import datetime
from PIL import Image

from mtkresearch.llm.image_process_v3 import load_image


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


class MRPromptBase:
    def generate_call_id(self):
        length = 24
        pool = string.ascii_letters + string.digits
        key = ''.join(random.choice(pool) for i in range(length))
        return f'call_{key}'

    def _check_arguments(self, arguments, func_description):
        errors = []

        if not isinstance(func_description['parameters'], dict) or func_description['parameters'] == {}:  # for the function which no need param.
            return
        
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
            elif expected_type == 'float' and not (isinstance(value, float) or isinstance(value, int)):
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
                if not (isinstance(conv['content'], str) or isinstance(conv['content'], list)):
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

                if not (isinstance(conv['content'], str) or isinstance(conv['content'], list)):
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
                    
            if isinstance(func['parameters'], dict) and 'properties' in func['parameters'].keys():
                for param, param_dict in func['parameters']['properties'].items():
                    if isinstance(param_dict, dict) and 'default' in param_dict.keys():
                        def parse_value(value_str, expected_type):
                            if expected_type == str:
                                return value_str
                            elif expected_type == int:
                                # 先轉換為 float，然後轉換為 int
                                if type(value_str) == str:
                                    raise ValueError("Expect int but get str")
                                return int(float(value_str))
                            elif expected_type == float:
                                return float(value_str)
                            elif expected_type == bool:
                                return value_str in ('true', 'yes', '1', 'on')
                            else:
                                # TODO
                                pass

                        type_map = {
                            'string': str,
                            'integer': int,
                            'float': float,
                            'boolean': bool,
                            # 'list': list,
                            # 'dict': dict
                        }

                        expected_type = type_map.get(param_dict['type'].lower())
                        parsed_value = parse_value(param_dict['default'], expected_type)
                        if expected_type in type_map.keys() and not isinstance(parsed_value, expected_type):
                            raise ValueError("Default value type mismatch")


class MRPromptV1(MRPromptBase):
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

    def get_prompt(self, conversations, add_bos_token=False, check=True):
        if check:
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


class MRPromptV2(MRPromptBase):
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

    def get_prompt(self, conversations, functions=None, add_bos_token=False, check=True):
        config = {
            'add_decision_token': True,
            'add_reason': False,
        }
        
        if check:
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


class MRPromptV3(MRPromptBase):
    '''prompt aligns to llama3.2'''

    def __init__(self, bos_token='<|begin_of_text|>', eos_token='<|end_of_text|>',
                 header_start_token='<|start_header_id|>', header_end_token='<|end_header_id|>',
                 turn_end_token='<|eot_id|>', message_end_token='<|eom_id|>',
                 tool_call_token='<|use_tool|>', answer_token='<|answer|>',
                 sys_role_token='system', user_role_token='user', assistant_role_token='assistant', 
                 tool_role_token='ipython', python_tag_token='<|python_tag|>',
                 image_start_token='<|start_img|>', image_end_token='<|end_img|>',
                 image_content_token='<|img|>',
                 bbox_start_token='<|start_bbox|>', bbox_end_token='<|end_bbox|>'
                ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.header_start_token = header_start_token
        self.header_end_token = header_end_token
        self.tool_call_token = tool_call_token
        self.answer_token = answer_token
        self.turn_end_token = turn_end_token
        self.message_end_token = message_end_token
        self.sys_role_token = sys_role_token
        self.user_role_token = user_role_token
        self.assistant_role_token = assistant_role_token
        self.tool_role_token = tool_role_token
        self.python_tag_token = python_tag_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.image_content_token = image_content_token
        self.bbox_start_token = bbox_start_token
        self.bbox_end_token = bbox_end_token

    def _get_sys_segment(self, sys=None, functions=None, training=False):
        if training:
            sys_content = sys.strip() if sys is not None else 'You are a helpful AI assistant.'
        else:
            formatted_date = datetime.now().strftime("%d %b %Y")
            sys_content = f'Cutting Knowledge Date: Oct 2024\nToday Date: {formatted_date}\n\n'
            sys_content += (sys.strip() if sys is not None else 'You are a helpful AI assistant built by MediaTek Research. The user you are helping speaks Traditional Chinese and comes from Taiwan.')

        functions_str = ''
        if functions:
            functions = repr(functions)
            functions_str = f'Customized Functions: {functions}\n\n---\n'

        return f'{self.header_start_token}{self.sys_role_token}{self.header_end_token}\n\n{functions_str}{sys_content}{self.turn_end_token}'

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _total_image_token(self, orig_size,
                          min_num=1,
                          max_num=12,
                          image_size=448,
                          use_thumbnail=True):
        orig_width, orig_height = orig_size

        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {(i, j)
                        for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if max_num >= i * j >= min_num}
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(aspect_ratio,
                                                        target_ratios, orig_width,
                                                        orig_height, image_size)
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        if use_thumbnail and blocks != 1 :
            blocks += 1

        return blocks

    def _get_image_content_token_num(self, width, height, token_per_patch=256):
        return int(self._total_image_token((width, height)) * token_per_patch)
    
    def _normalize_bbox(self, box, width, height, max_size=1000):
        x1, y1, x2, y2 = box
        normalized_box = [
            round((x1 / width) * max_size),
            round((y1 / height) * max_size),
            round((x2 / width) * max_size),
            round((y2 / height) * max_size),
        ]
        return normalized_box
    
    def _get_content_from_list(self, content_list):
        content = ''
        pixel_values = None
        for x in content_list:
            if x['type'] == 'text':
                content += x['text']
            elif x['type'] == 'image':
                image = Image.open(x['image_path']).convert('RGB')
                pixel_values = load_image(image)
                width, height = image.size
                image_content_token_num = self._get_image_content_token_num(width, height)
                image_content_str = ''.join([self.image_content_token] * image_content_token_num)
                content += f'{self.image_start_token}{image_content_str}{self.image_end_token}\n'
            elif x['type'] == 'bbox':
                bboxes = [self._normalize_bbox(box, x["width"], x["height"]) for box in x["coords"]]
                content += f'{self.bbox_start_token}{repr(bboxes)}{self.bbox_end_token}'
            else:
                raise ValueError('unknown type in the content')
        return content, pixel_values

    def _get_user_segment(self, user_content):
        if isinstance(user_content, list):
            user_content, pixel_values = self._get_content_from_list(user_content)
        else:
            pixel_values = None

        return (
            f'{self.header_start_token}{self.user_role_token}{self.header_end_token}\n\n{user_content}{self.turn_end_token}',
            pixel_values
        )

    def _get_assistant_prefix(self):
        return f'{self.header_start_token}{self.assistant_role_token}{self.header_end_token}\n\n'

    def _get_assistant_chat_completion(self, assistant_content, use_decision_token=False, is_end=False):
        if isinstance(assistant_content, list):
            assistant_content, pixel_values = self._get_content_from_list(assistant_content)
            assert pixel_values is None

        if is_end:
            prefix = self.answer_token if use_decision_token else ''
        else:
            prefix = ''
        return f'{prefix}{assistant_content}{self.turn_end_token}'
    
    def _get_assistant_call_completion(self, tool_calls, use_decision_token=False, is_end=False):
        call_instances = []
        for c in tool_calls:
            name = c["function"]["name"]
            argument_str = ', '.join(
                [f'{k}={repr(v)}' for k, v in json.loads(c["function"]["arguments"]).items()]
            )
            call_instances.append(f'{name}({argument_str})')
        tool_calls_str = '[' + ','.join(call_instances) + ']'

        if is_end:
            prefix = self.tool_call_token if use_decision_token else ''
            end = self.turn_end_token
        else:
            prefix = self.python_tag_token
            end = self.message_end_token

        return f'{prefix}{tool_calls_str}{end}'
    
    def _get_tool_response_segment(self, responses):
        return f'{self.header_start_token}{self.tool_role_token}{self.header_end_token}\n\n{repr(responses)}{self.turn_end_token}'

    def get_prompt(self, conversations, functions=None, add_bos_token=False, training=False, check=True):
        if check:
            if functions:
                self.check_functions(functions)
                self.check_conversations(conversations, functions=functions)
            else:
                self.check_conversations(conversations)
                
        config = {
            'add_decision_token': True,
        }
        if functions is None:
            config['add_decision_token'] = False

        prompt = self.bos_token if add_bos_token else ''

        sys = None
        if conversations[0]['role'] == 'system':
            sys = conversations[0]['content']
            conversations = conversations[1:]

        prompt += self._get_sys_segment(sys=sys, functions=functions, training=training)

        tmp_call_ids = []
        tmp_response_map = {}
        pixel_values = None

        for i, conv in enumerate(conversations):
            is_end = i + 1 >= len(conversations)

            if conv['role'] == 'user':
                user_segment, candidate_pixel_values = self._get_user_segment(conv["content"])
                if candidate_pixel_values is not None:
                    assert pixel_values is None, "only support single image in one conversation"
                    pixel_values = candidate_pixel_values
                prompt += user_segment
                prompt += self._get_assistant_prefix()

            elif conv['role'] == 'assistant' and 'tool_calls' not in conv:
                prompt += self._get_assistant_chat_completion(conv['content'], 
                                                              use_decision_token=config['add_decision_token'],
                                                              is_end=is_end)

            elif conv['role'] == 'assistant' and 'tool_calls' in conv:
                prompt += self._get_assistant_call_completion(conv['tool_calls'],
                                                              use_decision_token=config['add_decision_token'],
                                                              is_end=is_end)
                for c in conv['tool_calls']:
                    if 'id' in c:
                        tmp_call_ids.append(c['id'])
                        tmp_response_map[c['id']] = None

            elif conv['role'] == 'tool':
                assert conv['tool_call_id'] in tmp_call_ids
                tmp_response_map[conv['tool_call_id']] = json.loads(conv['content'])

                if i + 1 == len(conversations) or conversations[i + 1]['role'] != 'tool':
                    responses = [tmp_response_map[call_id] for call_id in tmp_call_ids]
                    prompt += self._get_tool_response_segment(responses)

                    tmp_call_ids = []
                    tmp_response_map = {}

                    prompt += self._get_assistant_prefix()

        if pixel_values is None:
            return prompt
        else:
            return (prompt, pixel_values)

    def _replace_function_name(self, content):
        def _parenthetic_contents(string, level=0):
            """Generate parenthesized contents in string as pairs (level, contents)."""
            stack = []
            for i, c in enumerate(string):
                if c == '(':
                    stack.append(i)
                elif c == ')' and stack:
                    start = stack.pop()
                    now_level = len(stack)
                    if now_level == level:
                        yield (start, i)

        replaced_map = {}
        i = 1
        replaced_str = '['
        for start, end in _parenthetic_contents(content):
            replaced_name = 'F' + str(len(replaced_map))
            name = content[i:start].lstrip()
            common_delta = content[end:].index(',') + 1 if ',' in content[end:] else 0
            i = end + common_delta
            replaced_str += replaced_name + content[start:end + common_delta] 
            replaced_map[replaced_name] = name
        replaced_str += ')]'
        return replaced_str, replaced_map

    def parse_generated_str(self, generated_str):
        generated_str = generated_str.strip()
        generated_str = _removeprefix(generated_str, self.answer_token).strip()
        generated_str = _removesuffix(generated_str, self.turn_end_token).strip()

        if self.tool_call_token in generated_str: # function call
            generated_str = _removeprefix(generated_str, self.tool_call_token).strip()
            
            try:
                if not generated_str.startswith('['):
                    raise ValueError("Wrong format")
                replaced_str, replace_map = self._replace_function_name(generated_str)

                tree = ast.parse(replaced_str, mode='eval')
                
                # Ensure the root node is a list
                if not isinstance(tree.body, ast.List):
                    raise ValueError("Input string must be a list of function calls")
                
                function_calls = [
                    {
                        'name': replace_map[node.func.id],
                        'arguments': json.dumps({
                            keyword.arg: ast.literal_eval(keyword.value)
                            for keyword in node.keywords
                        }, ensure_ascii=False)
                    } 
                    for node in tree.body.elts
                ]

                conv = {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': self.generate_call_id(),
                            'type': 'function',
                            'function': func_call
                        }
                        for func_call in function_calls
                    ]
                }
            except Exception as e:
                conv = {
                    'role': 'assistant',
                    'content': generated_str
                }
        else:
            conv = {
                'role': 'assistant',
                'content': generated_str
            }
        return conv
