from typing import List, Dict, Optional
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from utils import system_prompt

class LuminariaChatService(OpenAIChatCompletion):
  # System message to be included in every request
  _SYSTEM_MESSAGE = {
    'role': 'system',
    'content': system_prompt.strip()
  }

  async def _send_request(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **kwargs):
    # Include the system message in every request
    messages_with_system_message = [self._SYSTEM_MESSAGE] + (messages or [])
    if prompt:
      messages_with_system_message.append({'role': 'user', 'content': prompt})
    # Send the request with the system message included
    return await super()._send_request(messages=messages_with_system_message, **kwargs)

