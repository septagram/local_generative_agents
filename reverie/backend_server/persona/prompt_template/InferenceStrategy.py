import os
import re
import traceback

from typing import Any, Dict, List, Union, Optional, TypeVar
from operator import attrgetter
from enum import Enum, auto
from asyncio import get_event_loop
from termcolor import colored
from termcolor._types import Color
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_core.runnables import chain, Runnable, RunnableLambda
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.exceptions import OutputParserException
from langchain_community.vectorstores import Chroma
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

import utils as config
from persona.prompt_template.SimplifiedPedanticOutputParser import SimplifiedPydanticOutputParser, JSONType, find_and_parse_json
from persona.prompt_template.embedding import LocalEmbeddings
from persona.common import deindent

if config.debug_cache_clear and os.path.exists(".langchain.db"):
  os.remove(".langchain.db")
if config.debug_cache_enabled:
  set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def ColorEcho(color: Optional[Color] = None, template: str = "{value}", full_prompt: bool = False):
  def invokeColorEcho(value: Union[ChatPromptValue, BaseMessage, str]):
    displayValue = value
    if not full_prompt:
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
  return ChatPromptTemplate(
    messages=[
      HumanMessagePromptTemplate.from_template(deindent(prompt))
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
    wrap_prompt(prompt.replace("{", "{{").replace("}", "}}")) |
    ColorEcho('light_blue') |
    add_system_prompt |
    model |
    RunnableLambda(attrgetter('content')) |
    ColorEcho('cyan')
  )
  return lambda: chain.invoke({"prompt": prompt})

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
          messages = (
            (current_prompt.messages if config.do_retry_with_full_history else prompt.messages) +
            [output] +
            wrap_prompt(retry_prompt).format_messages(error=error)
          )
        )
    raise OutputParserException("Out of retries")
  return chain_with_retries

class NoExampleSelector(BaseExampleSelector):
    def add_example(self, example: Dict[str, str]) -> Any:
      pass
    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
      return []

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
  output_type: Union[type, OutputType] = OutputType.Text
  prompt: Optional[str] = None
  example_prompt: Optional[str] = ""
  config: Dict[str, Any] = {}
  context: Dict[str, Any] = {}
  examples: List[Dict[str, Any]] = []
  example_count: int = 3
  example_selector: BaseExampleSelector = NoExampleSelector()

  def __init__(self):
    output_parser = self.output_parser()
    format_instructions = (
      output_parser.get_format_instructions()
      if callable(getattr(output_parser, 'get_format_instructions', None))
      else None
    )

    @chain
    def prepare_context(args: List):
      self.context = self.prepare_context(*args)
      if format_instructions:
        self.context['format_instructions'] = format_instructions
      return self.context

    if self.examples:
      self.example_selector = SemanticSimilarityExampleSelector.from_examples(
        [
          {
            "input": deindent(self.example_prompt).format(**example),
            "output": example['output'],
          }
          for example in self.examples
        ],
        LocalEmbeddings(),
        Chroma,
        k=self.example_count,
      )

    @chain
    def fetch_examples(context: Dict[str, Any]):
      if self.example_selector:
        search_object = {"input": deindent(self.example_prompt).format(**context)}
        selected_examples = self.example_selector.select_examples(search_object)
        context["examples"] = '\n\n'.join([
          ('{input}\nAnswer: {output}').format(**example)
          for example in selected_examples
        ])
        context["example_prompt"] = deindent(self.example_prompt).format(**context)
      return context

    self.chain = (
      announcer(self.__class__.__name__) |
      prepare_context |
      fetch_examples |
      wrap_prompt(self.prompt) |
      with_retries(
        retries=self.retries,
        inference_chain=(
          ColorEcho('light_blue') |
          add_system_prompt |
          model(ModelAlias.strong, self.config) |
          ColorEcho('cyan')
        ),
        output_parser_chain=(
          output_parser |
          RunnableLambda(lambda result: self.postprocess(result)) |
          ColorEcho('light_green', "Final output: {value}")
        ),
      )
    )

  def prepare_context(self, *args: ArgsType) -> Dict[str, str]:
    return {}
  
  def output_parser(self) -> BaseOutputParser:
    if isinstance(self.output_type, OutputType):
      @chain
      def validate(llm_output: str) -> Union[JSONType, str]:
        if self.output_type == OutputType.JSON:
          json_output = find_and_parse_json(llm_output.content)
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
    else:
      return SimplifiedPydanticOutputParser(pydantic_object=self.output_type)
  
  def validate_text(self, output: str) -> Optional[str]:
    return None
  
  def validate_json(self, json: JSONType) -> Optional[str]:
    return None

  def extract_text(self, output: str) -> ReturnType:
    return output
  
  def extract_json(self, json: JSONType) -> ReturnType:
    return json
  
  def postprocess(self, result: Any):
    return result
  
  def fallback(self, *args: ArgsType) -> ReturnType:
    raise ValueError("LLM output didn't pass validation with no fallback function defined.")
  
  def __call__(self, *args: ArgsType) -> ReturnType:
    return self.chain.invoke(args)
