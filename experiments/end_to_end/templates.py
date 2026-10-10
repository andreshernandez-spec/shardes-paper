"""Chat templates, copied verbatim from open-instruct's CHAT_TEMPLATES.

Extracted with `ast.literal_eval` from `open_instruct/dataset_transformation.py`, not
retyped: `olmo_thinker` at d928a7c (the Olmo 3 RL-Zero IF run), `tulu` at 3f37c29 (Tulu
3.1; identical at d928a7c). The sha256 of each string is pinned so an edit shows up as a
failing test rather than as a silently different prompt.

`olmo_thinker` ends the generation prompt with `<think>`, and the run stopped on
`</answer>`; both matter for response lengths.
"""

OLMO_THINKER = "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}{% if not has_system %}{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}{% endif %}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] }}{% if message.get('functions', none) is not none %}{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}{% else %}{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}{% endif %}{% elif message['role'] == 'user' %}{% if message.get('functions', none) is not none %}{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}{% else %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% if message.get('content', none) is not none %}{{ message['content'] }}{% endif %}{% if message.get('function_calls', none) is not none %}{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}{% endif %}{% if not loop.last %}{{ '<|im_end|>' + '\n' }}{% else %}{{ eos_token }}{% endif %}{% elif message['role'] == 'environment' %}{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n<think>' }}{% endif %}{% endfor %}"
OLMO_THINKER_SHA256 = "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a"

TULU = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
TULU_SHA256 = "ac7498a36a719da630e99d48e6ebc4409de85a77556c2b6159eeb735bcbd11df"
