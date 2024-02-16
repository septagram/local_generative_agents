from typing import List, Dict, Optional
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from utils import system_prompt
from termcolor import colored

from persona.prompt_template.DebugCache import DebugCache

class LuminariaChatService(OpenAIChatCompletion):
  # System message to be included in every request
  _SYSTEM_MESSAGE = {
    'role': 'assistant', #'system',
    'content': system_prompt.strip()
  }
  # Yet we set the role to "assistant", because text-generation-webui ignores system message,
  # at least for Mistral Instruct.
  #
  # See https://tree.taiga.io/project/septagram-ai-town/issue/82

  async def _send_request(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **kwargs):
    # Include the system message in every request
    messages_with_system_message = [self._SYSTEM_MESSAGE] + (messages or [])
    if prompt:
      messages_with_system_message.append({'role': 'user', 'content': prompt})
    print(colored(prompt, 'light_blue'))
    # Send the request with the system message included
    response = await DebugCache.read(prompt) or await super()._send_request(messages=messages_with_system_message, **kwargs)
    DebugCache.write(prompt, response)
    return response
