import json


class MRChatManager:
    def __init__(self, prompt, sys_prompt=None, functions=None):
        self.functions = functions
        self.prompt = prompt  # MRPromptV1, MRPromptV2

        if sys_prompt:
            self.conversations = [
                {'role': 'system', 'content': sys_prompt}
            ]
        else:
            self.conversations = []
        self._last_func_calls = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def user_input(self, message):
        if self._last_func_calls:
            raise ValueError
        self.conversations.append(
            {
                'role': 'user',
                'content': message
            }
        )

    def func_response(self, call_id, result):
        if self.functions is None:
            raise ValueError
        if not isinstance(result, dict):
            raise ValueError

        name = self._last_func_calls[call_id]['name']
        self.conversations.append(
            {
                'role': 'tool',
                'tool_call_id': call_id,
                'name': name,
                'content': json.dumps(result)
            }
        )
        del self._last_func_calls[call_id]

    def parse_assistant(self, generated_str):
        if self._last_func_calls:
            raise ValueError

        conv = self.prompt.parse_generated_str(generated_str)
        self.conversations.append(conv)

        if 'tool_calls' in conv:
            self._last_func_calls = {x['id']: x['function'] for x in conv['tool_calls']}
            return {'func_calls':[
                {
                    'name': x['function']['name'],
                    'arguments': json.loads(x['function']['arguments']),
                    'id': x['id']
                } for x in conv['tool_calls']
            ]}
        else:
            return {'message': conv['content']}
