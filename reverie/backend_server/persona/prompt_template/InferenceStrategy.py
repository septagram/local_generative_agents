import re
import traceback
import json

from typing import Any, Dict, List, Union, Optional, TypeVar
from operator import attrgetter
from enum import Enum, auto
from asyncio import get_event_loop
from termcolor import colored
from termcolor._types import Color
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain_core.runnables import chain, Runnable, RunnableLambda
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import RetryWithErrorOutputParser

import utils as config

def ColorEcho(color: Optional[Color] = None, template: str = "{value}"):
  def invokeColorEcho(value: Union[ChatPromptValue, BaseMessage, str]):
    displayValue = value
    if isinstance(displayValue, ChatPromptValue):
      displayValue = displayValue.messages[-1]
    if isinstance(displayValue, BaseMessage):
      displayValue = displayValue.content
    print(colored(template.format(value=displayValue), color), flush=True)
    return value
  return RunnableLambda(invokeColorEcho)

class DeprecatedOverrideTypes(Enum):
  GPT4 = "gpt4"

class ModelAlias(Enum):
  strong = 'inference_model_strong'
  superstrong = 'inference_model_superstrong'

def model(alias: ModelAlias, prompt_config: Dict[str, Union[str, float]]):
  return ChatOpenAI(
    cache=None,
    base_url=config.openai_api_base if alias != ModelAlias.superstrong else None,
    openai_api_key=config.openai_api_key,
    model=getattr(config, alias.value),
    temperature=prompt_config.get("temperature", 0.5),
    max_tokens=prompt_config.get("max_tokens", 500),
    # top_k=prompt_config.get("top_k"),
    # top_p=prompt_config.get("top_p"),
    # min_p=prompt_config.get("min_p"),
    # frequency_penalty=prompt_config.get("frequency_penalty"),
    # presence_penalty=prompt_config.get("presence_penalty"),
    # stop_sequences=prompt_config.get("stop"),
  )

def announcer(name):
  def announce(args):
    curr_time = re.search(r'\b(\d\d:\d\d):\d\d$', str(get_event_loop().reverie_server.curr_time)).group(1)
    print(
      ' '.join([
        f"[{colored(curr_time, 'dark_grey')}]",
        "Semantic function:",
        colored(name, 'yellow'),
      ])
    )
    return args
  return RunnableLambda(announce)

def wrap_prompt(prompt: str):
  # Deindent by removing the first sequence of spaces or tabs starting with \n
  # and removing all its instances
  match = re.search(r'\n[ \t]+', prompt)
  if match:
    first_indentation = match.group()
  else:
    first_indentation = '\n'
  deindented_prompt = prompt.strip().replace(first_indentation, '\n')

  return ChatPromptTemplate(
    messages=[
      HumanMessagePromptTemplate.from_template(deindented_prompt)
    ]
  )

def prepend_prompt(messages: List[BaseMessage]):
  return RunnableLambda(lambda prompt: ChatPromptValue(messages=messages + prompt.messages))

add_system_prompt = prepend_prompt([
  AIMessage(content=config.system_prompt)
])

@chain
def add_system_prompt(prompt):
  system_prompt = AIMessage(content=config.system_prompt)
  # Should be SystemMessage, but text-generation-webui doesn't know what to do with it for Mistral
  return ChatPromptValue(messages=[system_prompt] + prompt.messages)

def inline_semantic_function(function_name: str, prompt_config: Dict[str, Any], prompt: str, use_openai=False):
  model = ChatOpenAI(
    cache=None,
    base_url=config.openai_api_base if not use_openai else None,
    openai_api_key=config.openai_api_key,
    model=config.inference_model_superstrong if use_openai else config.inference_model_strong,
    temperature=prompt_config.get("temperature", 0.5),
    max_tokens=prompt_config.get("max_tokens", 500),
    # top_k=prompt_config.get("top_k", 0),
    # top_p=prompt_config.get("top_p", 0.95),
    # min_p=prompt_config.get("min_p", 0.05),
    # frequency_penalty=prompt_config.get("frequency_penalty", 0),
    # presence_penalty=prompt_config.get("presence_penalty", 0),
    # stop_sequences=prompt_config.get("stop", None),
  )
  chain = (
    announcer("LEGACY_" + function_name) |
    wrap_prompt("{prompt}") |
    ColorEcho('light_blue') |
    add_system_prompt |
    model |
    RunnableLambda(attrgetter('content')) |
    ColorEcho('cyan')
  )
  return lambda: chain.invoke({"prompt": prompt})
  
JSONType = Union[Dict[str, Any], List[Any]]

def find_and_parse_json(message: AIMessage) -> JSONType:
  text = message.content
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

def with_retries(retries: int, inference_chain: Runnable, output_parser_chain: Runnable):
  # retry_chain = 
  retry_prompt = """
      This reply has the following issue:

      {error}

      Let's correct our reply.
  """

  @chain
  def chain_with_retries(prompt: ChatPromptValue):
    current_prompt = prompt

    for _ in range(retries + 1):
      output = inference_chain.invoke(current_prompt)
      try:
        return output_parser_chain.invoke(output)
      except Exception as error:
        current_prompt = ChatPromptValue(
          messages = prompt.messages + [
            output,
            HumanMessage(content=retry_prompt.format(error=error))
          ]
        )
    raise OutputParserException("Out of retries")
  return chain_with_retries

def functor(cls):
  instance = cls()
  def infer(*args: ArgsType) -> ReturnType:
    return instance(*args)
  return infer

class OutputType(Enum):
  Text = auto()
  JSON = auto()

class InferenceStrategy:
  retries: int = 5
  output_type: OutputType = OutputType.Text
  prompt: Optional[str] = None
  config: Dict[str, Any] = {}
  context: Dict[str, Any] = {}

  def __init__(self):
    @chain
    def prepare_context(args: List):
      self.context = self.prepare_context(*args)
      return self.context

    self.chain = (
      announcer(self.__class__.__name__) |
      prepare_context |
      wrap_prompt(self.prompt) |
      with_retries(
        retries=self.retries,
        inference_chain=(
          ColorEcho('light_blue') | #, '{value.messages[-1].content}') |
          add_system_prompt |
          model(ModelAlias.strong, self.config) |
          ColorEcho('cyan') #, "{value.content}")
        ),
        output_parser_chain=self.output_parser(),
      ) |
      ColorEcho('light_green', "Final output: {value}")
    )

  def prepare_context(self, *args: ArgsType) -> Dict[str, str]:
    return {}
  
  def output_parser(self) -> BaseOutputParser:
    @chain
    def validate(llm_output: str) -> Union[JSONType, str]:
      if self.output_type == OutputType.JSON:
        json_output = find_and_parse_json(llm_output)
        validation_error = self.validate_json(json_output)
      elif self.output_type == OutputType.Text:
        validation_error = self.validate_text(llm_output)
      else:
        raise ValueError(f"Unknown output type: {self.output_type}")
      if validation_error is not None:
        print(colored(f"Error in interaction with {self.__class__.__name__}: {str(validation_error)}", 'red'))
        if '__traceback__' in validation_error:
          traceback.print_exception(type(validation_error), validation_error, validation_error.__traceback__)
        if config.strict_errors:
          raise ValueError(validation_error)
        else:
          return self.fallback
      return json_output or llm_output

    extract = RunnableLambda(
      self.extract_json
      if self.output_type == OutputType.JSON
      else self.extract_text
    )

    return validate | extract
  
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
    return self.chain.invoke(args)
