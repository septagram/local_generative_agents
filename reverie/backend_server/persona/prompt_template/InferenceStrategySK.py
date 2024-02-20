import re
import traceback
import json
import openai
import semantic_kernel as sk

from typing import Any, Dict, List, Union, Optional, TypeVar
from enum import Enum, auto
from openai import AsyncOpenAI
from asyncio import get_event_loop
from termcolor import colored
from semantic_kernel.orchestration.sk_function import SKFunction

from persona.prompt_template.LuminariaChatService import LuminariaChatService

import utils as config

class DeprecatedOverrideTypes(Enum):
  GPT4 = "gpt4"

openai.api_key = config.openai_api_key
if hasattr(config, 'openai_api_base'):
    openai.base_url = config.openai_api_base

base_url = config.openai_api_base if hasattr(config, 'openai_api_base') else None

kernel = sk.Kernel()
client_local = AsyncOpenAI(api_key=config.openai_api_key, base_url=base_url)
client_openai = AsyncOpenAI(api_key=config.openai_api_key)
kernel.add_chat_service("strong", LuminariaChatService(config.inference_model_strong, async_client=client_local))
kernel.add_chat_service("superstrong", LuminariaChatService(config.inference_model_superstrong, async_client=client_openai))

def inline_semantic_function(function_name: str, config: Dict[str, Any], prompt: str, use_openai=False):
  # Find the first sequence of spaces or tabs starting with \n
  match = re.search(r'\n[ \t]+', prompt)
  if match:
    first_indentation = match.group()
  else:
    first_indentation = '\n'
  prompt = prompt.strip()
  prompt = prompt.replace(first_indentation, '\n')
  kernel.set_default_chat_service('superstrong' if use_openai else 'strong')
  kernel.set_default_text_completion_service('superstrong' if use_openai else 'strong')
  return kernel.create_semantic_function(
    prompt_template=prompt,
    skill_name="simulation",
    function_name=function_name + "_gpt4" if use_openai else function_name,
    temperature=config.get("temperature", 0.5),
    max_tokens=config.get("max_tokens", 500),
    # top_k=config.get("top_k", 0),
    top_p=config.get("top_p", 0.95),
    # min_p=config.get("min_p", 0.05),
    frequency_penalty=config.get("frequency_penalty", 0),
    presence_penalty=config.get("presence_penalty", 0),
    stop_sequences=config.get("stop", None),
  )

JSONType = Union[Dict[str, Any], List[Any]]

def find_and_parse_json(text: str) -> JSONType:
  OpeningToClosingBrace = {"{": "}", "[": "]"}
  start_indices = [text.find('{'), text.find('[')]
  start_index = min(i for i in start_indices if i != -1)

  if start_index == -1:
    return None
  
  closing_brace = OpeningToClosingBrace[text[start_index]]
  text = text[start_index:]
  last_error = ValueError("Expected JSON object/array")

  while True:
    end_index = text.rfind(closing_brace)
    if end_index == -1:
      raise last_error
    try:
      return json.loads(text[:end_index+1])
    except json.JSONDecodeError as e:
      last_error = e
      text = text[:end_index]

# Define type variables
ArgsType = TypeVar('ArgsType')  # The type of the arguments
ReturnType = TypeVar('ReturnType')  # The type of the return value

def functor(cls):
  def infer(*args: ArgsType) -> ReturnType:
    return cls()(*args)
  return infer

class OutputType(Enum):
  Text = auto()
  JSON = auto()

class InferenceStrategySK:
  semantic_function: SKFunction = kernel.create_semantic_function(
    prompt_template="Don't answer this request. Output nothing.",
    function_name='no_reply',
  )
  semantic_function_gpt4: Optional[SKFunction] = None
  retries: int = 5
  output_type: OutputType = OutputType.Text
  prompt: Optional[str] = None
  config: Dict[str, Any] = {}

  def __init__(self):
    if self.semantic_function and self.semantic_function != InferenceStrategySK.semantic_function:
      return
    if isinstance(self.prompt, str):
      self.semantic_function = inline_semantic_function(self.__class__.__name__, self.config, self.prompt)
      self.semantic_function_gpt4 = inline_semantic_function(self.__class__.__name__, self.config, self.prompt, use_openai=True)

  def prepare_context(self, *args: ArgsType) -> Dict[str, str]:
    return {}
  
  def validate_text(self, output: str) -> Optional[str]:
    return None
  
  def validate_json(self, json: JSONType) -> Optional[str]:
    return None

  def extract_text(self, output: str) -> ReturnType:
    return output
  
  def extract_json(self, json: JSONType) -> ReturnType:
    return json
  
  def fallback(self, *args: ArgsType) -> ReturnType:
    raise ValueError("LLM output didn't pass validation with no fallback function defined.")
  
  def __call__(self, *args: ArgsType) -> ReturnType:
    # Step 1: Prepare the context
    context = kernel.create_new_context()
    self.context_variables = self.prepare_context(*args)
    for key, value in self.context_variables.items():
      context[key] = str(value)

    llm_output = final_output = last_error = None

    for i in range(self.retries):
      llm_output = ""
      final_output = None
      last_error = None
      
      try:
        curr_time = re.search(r'\b(\d\d:\d\d):\d\d$', str(get_event_loop().reverie_server.curr_time)).group(1)
        retry_string = "" if i == 0 else f" Retry: {i}"
        print(
          ' '.join([
            f"[{colored(curr_time, 'dark_grey')}{colored(retry_string, 'red')}]",
            "Semantic function:",
            colored(self.semantic_function.name, 'yellow'),
          ])
        )
        # if i == 0:
        #   # Output the full prompt. (How do we do that in SK?)
        #   prompt = self.semantic_function._chat_prompt_template
        #   for key, value in context_dict.items():
        #     prompt = prompt.replace("{{$" + key + "}}", str(value))
        #   print(colored(prompt, 'blue'))

        # Step 3: Invoke the semantic function
        if i == self.retries - 1 and self.semantic_function_gpt4 is not None:
          llm_output = str(self.semantic_function_gpt4(context=context)).strip()
        else:
          llm_output = str(self.semantic_function(context=context)).strip()
        print(colored(llm_output, 'cyan'))
        json_output = None
        if self.output_type == OutputType.JSON:
          json_output = find_and_parse_json(llm_output)

        # Step 4: Validate the output
        validation_error = None
        if self.output_type == OutputType.JSON:
          validation_error = self.validate_json(json_output)
        elif self.output_type == OutputType.Text:
          validation_error = self.validate_text(llm_output)
        else:
          raise ValueError(f"Unknown output type: {self.output_type}")
        if validation_error is not None:
          raise ValueError(validation_error)

        # Step 5: Extract the data from the output
        if self.output_type == OutputType.JSON:
          final_output = self.extract_json(json_output)
        elif self.output_type == OutputType.Text:
          final_output = self.extract_text(llm_output)

        # Successful, so break the loop
        break

      except Exception as e:
        last_error = e
        print(colored(f"Error in interaction with {self.semantic_function.name}: {str(last_error)}", 'red'))
        traceback.print_exception(type(last_error), last_error, last_error.__traceback__)

    if last_error is None:
      if final_output != llm_output:
        print(colored(f"Final output: {final_output}", 'light_green'))
      return final_output
    else:
      if config.strict_errors:
        raise last_error
      else:
        return self.fallback(*args)
