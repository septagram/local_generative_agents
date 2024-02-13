"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time
import os
import re
import semantic_kernel as sk
import traceback

from typing import Any, Dict, List, Union, Optional, TypeVar
from enum import Enum, auto
from openai import AsyncOpenAI
from asyncio import get_event_loop
from termcolor import colored
from semantic_kernel.orchestration.sk_function import SKFunction

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from persona.prompt_template.LuminariaChatService import LuminariaChatService

from utils import *

class DeprecatedOverrideTypes(Enum):
  GPT4 = "gpt4"

openai.api_key = openai_api_key
if 'openai_api_base' in globals():
    openai.base_url = openai_api_base

base_url = openai_api_base if 'openai_api_base' in globals() else None

kernel = sk.Kernel()
client_local = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
client_openai = AsyncOpenAI(api_key=openai_api_key)
kernel.add_chat_service("strong", LuminariaChatService(inference_model_strong, async_client=client_local))
kernel.add_chat_service("superstrong", LuminariaChatService(inference_model_superstrong, async_client=client_openai))

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
      return json.loads(text[start_index:end_index+1])
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

class InferenceStrategy:
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
    if self.semantic_function and self.semantic_function != InferenceStrategy.semantic_function:
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
        json_output = None
        if self.output_type == OutputType.JSON:
          json_output = find_and_parse_json(llm_output)
        print(colored(llm_output, 'cyan'))

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

    if last_error is None:
      if final_output != llm_output:
        print(colored(f"Final output: {final_output}", 'light_green'))
      return final_output
    else:
      print(colored(f"Error in interaction with {self.semantic_function.name}: {str(last_error)}", 'red'))
      if strict_errors:
        raise last_error
      else:
        return self.fallback(*args)

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def deprecated(prompt, gpt_parameter, override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override):
  full_prompt = f"Without any prelude, deliberation or reasoning, continue the following block of text:\n\n{prompt}"
  full_prompt_gpt4 = f"Without any prelude, deliberation or reasoning, continue the following block of text, the same way that GPT-3 would continue it (starting from the very next symbol):\n\n{prompt}"
  
  stack_trace = traceback.format_stack()
  function_name = None
  
  for trace in stack_trace:
    match = re.search(r'\brun_gpt_prompt_\w+', trace)
    if match:
      function_name = match.group()
      break
  
  if function_name is None:
    function_name = re.sub(r'[^\w\d]+', '_', '_'.join(stack_trace))

  kernel.set_default_chat_service('superstrong' if override_deprecated == DeprecatedOverrideTypes.GPT4 else 'strong')
  kernel.set_default_text_completion_service('superstrong' if override_deprecated == DeprecatedOverrideTypes.GPT4 else 'strong')
  class LegacyPrompt(InferenceStrategy):
    semantic_function = kernel.create_semantic_function(
      prompt_template=full_prompt_gpt4 if override_deprecated == DeprecatedOverrideTypes.GPT4 else full_prompt,
      function_name=function_name + "_gpt4" if override_deprecated == DeprecatedOverrideTypes.GPT4 else function_name,
      temperature=gpt_parameter.get("temperature", 0),
      max_tokens=gpt_parameter.get("max_tokens", 500),
      top_p=gpt_parameter.get("top_p", 1),
      frequency_penalty=gpt_parameter.get("frequency_penalty", 0),
      presence_penalty=gpt_parameter.get("presence_penalty", 0),
      stop_sequences=gpt_parameter.get("stop", None),
    )
    retries = 1

  output = str(LegacyPrompt()())
  if not override_deprecated:
    traceback.print_stack()
    print(colored(f"The function {function_name} is deprecated and will be removed.", 'red'))
    exit()
  return output

def ChatGPT_single_request(prompt, override_deprecated = inference_deprecated_override): 
  temp_sleep()
  return deprecated(prompt, {}, override_deprecated)

  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt, override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  return deprecated(prompt, {}, override_deprecated)

  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-4", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt, override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  return deprecated(prompt, {}, override_deprecated)
  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt, override_deprecated).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      if i == repeat - 1:
        curr_gpt_response = ChatGPT_request(prompt, DeprecatedOverrideTypes.GPT4).strip()
      else:
        curr_gpt_response = ChatGPT_request(prompt, override_deprecated).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt, override_deprecated).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter, override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  return deprecated(prompt, gpt_parameter, override_deprecated)
  try: 
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           override_deprecated: Union[bool, DeprecatedOverrideTypes] = inference_deprecated_override): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = None
    if i == repeat - 1:
      curr_gpt_response = GPT_request(prompt, gpt_parameter, override_deprecated=DeprecatedOverrideTypes.GPT4)
    else:
      curr_gpt_response = GPT_request(prompt, gpt_parameter, override_deprecated=override_deprecated)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_openai_embedding(text, model=embedding_model):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)
