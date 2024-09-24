
from mtkresearch.llm.chat import MRChatManager
from mtkresearch.llm.prompt import MRPromptV1, MRPromptV2


class TestMRChatManager:

    def test_normal_chat(self):
        prompt = MRPromptV1()
        with MRChatManager(prompt=prompt, sys_prompt='SYS') as manager:
            manager.user_input('Q1')
            manager.parse_assistant(' A1</s>')
            manager.user_input('Q2')

            assert manager.conversations == [
                {
                    'role': 'system',
                    'content': 'SYS'
                },
                {
                    'role': 'user',
                    'content': 'Q1'
                },
                {
                    'role': 'assistant',
                    'content': 'A1'
                },
                {
                    'role': 'user',
                    'content': 'Q2'
                },
            ]

    def test_tool_chat(self):
        functions = [
            {
                'name': 'F',
                'description': 'F-D',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'V1': {
                            'type': 'string',
                            'description': 'V1-D'
                        },
                        'V2': {
                            'type': 'string',
                            'enum': ['E1', 'E2']
                        }
                    },
                    'required': ['E1']
                }
            }
        ]
        prompt = MRPromptV2()
        with MRChatManager(prompt=prompt, sys_prompt='SYS', functions=functions) as manager:
            manager.user_input('Q1')
            
            result = manager.parse_assistant('<|use_tool|><|tool_call_begin|>{"name": "F", "arguments": "{\\"E1\\": \\"A1\\"}"}<|tool_call_end|><|tool_call_begin|>{"name": "F", "arguments": "{\\"E1\\": \\"A2\\"}"}<|tool_call_end|><|im_end|>')
            ids = []
            for x in result['func_calls']:
                ids.append(x['id'])
                manager.func_response(x['id'], {'result': f'R-{x["id"][-3:]}'})
            manager.parse_assistant(' A3<|im_end|>')
            manager.user_input('Q2')

            print(manager.conversations)
            assert manager.conversations == [
                {
                    'role': 'system',
                    'content': 'SYS'
                },
                {
                    'role': 'user',
                    'content': 'Q1'
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            'id': ids[0],
                            'type': 'function',
                            'function': {
                                'arguments': "{\"E1\": \"A1\"}",
                                'name': 'F'
                            }
                        },
                        {
                            'id': ids[1],
                            'type': 'function',
                            'function': {
                                'arguments': "{\"E1\": \"A2\"}",
                                'name': 'F'
                            }
                        },
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": ids[0],
                    "name": "F",
                    "content": f'{{"result": "R-{ids[0][-3:]}"}}'
                },
                {
                    "role": "tool",
                    "tool_call_id": ids[1],
                    "name": "F",
                    "content": f'{{"result": "R-{ids[1][-3:]}"}}'
                },
                {
                    'role': 'assistant',
                    'content': 'A3'
                },
                {
                    'role': 'user',
                    'content': 'Q2'
                },
            ]
