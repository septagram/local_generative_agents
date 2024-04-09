"""
 Copyright 2024 Joon Sung Park, Igor Novikov

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from __future__ import annotations

from emoji import is_emoji
from typing import TYPE_CHECKING, Any, Dict, Optional
from langchain_core.pydantic_v1 import Field, validator

from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy
from persona.prompt_template.ResponseModel import ResponseModel

if TYPE_CHECKING:
  from persona.persona import Persona

class PronunciatioResponse(ResponseModel):
  emojis: str = Field(..., description="one or two emojis representing the given action, together without any separator", min_length=1, max_length=100)
  firstEmoji: str = Field(..., description="the first emoji, exactly one", min_length=1, max_length=50)
  secondEmoji: Optional[str] = Field(description="the second emoji, exactly one, if appropriate", min_length=1, max_length=50)

  @validator('firstEmoji', 'secondEmoji')
  def is_emoji(cls, emoji: str):
    if not is_emoji(emoji):
      raise ValueError(f'"{emoji}" is not a valid emoji')
    return emoji
  
  @validator('secondEmoji')
  def not_same(cls, secondEmoji: str, values: Dict[str, Any]):
    if values.get('firstEmoji') == secondEmoji:
      raise ValueError(f'Emojis are the same. Use a different secondEmoji or omit it.')
    return secondEmoji

"""
Given an action description, creates an emoji string description via a few
shot (TODO) prompt. 

Does not really need any information from persona. TODO Except skin color and gender, perhaps? Because there's a rich variety of non-yellow-gender-neutral emojis.

INPUT: 
  act_desp: the description of the action (e.g., "sleeping")
  persona: The Persona class instance
OUTPUT: 
  a string of emoji that translates action description.
EXAMPLE OUTPUT: 
  "🧈🍞"
"""
@functor
class run_gpt_prompt_pronunciatio(InferenceStrategy):
  output_type = PronunciatioResponse
  config = {
    "temperature": 0.7
  }
  prompt = """
    We need to convert an action description into one or two emojis that best illustrate that action.

    {format_instructions}

    Action: {action_description}

    Which emojis would best represent it?
  """

  def prepare_context(self, action_description: str, persona: 'Persona'):
    return {
      'action_description': action_description
    }
  
  def postprocess(self, response: PronunciatioResponse):
    return response.firstEmoji + (response.secondEmoji or '')
