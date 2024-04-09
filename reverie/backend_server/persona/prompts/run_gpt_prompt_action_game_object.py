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

import json
from typing import Any, Dict, TYPE_CHECKING
from langchain_core.pydantic_v1 import Field, validator

from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy
from persona.prompt_template.ResponseModel import ResponseModel
from persona.common import with_json

if TYPE_CHECKING:
  from persona.persona import Persona

class ActionGameObjectResponse(ResponseModel):
  game_object: str = Field(..., description="the Game Object this character should use or go to")

  @validator('game_object')
  def validate_game_object(cls, game_object: str, values: Dict[str, Any]):
    if values['context']:
      if game_object not in values['context']['all_game_objects']:
        available_objects = f"Select one of {values['context']['all_game_objects']}, verbatim."
        raise ValueError(f"The \"{game_object}\" is not available to {values['context']['firstname']}. {available_objects}")
    return game_object

@functor
class run_gpt_prompt_action_game_object(InferenceStrategy):
  output_type = ActionGameObjectResponse
  config = {
    "temperature": 0
  }
  example_prompt = """
    Current activity: {action_description_json}
    Objects available: {all_game_objects_json}
    Pick ONE most relevant object from the objects available.
  """
  prompt = """
    We need to choose an appropriate Object for the task ahead.

    * A list of available objects will be given. These are the only objects to choose from.
    * If none of those fit very well, we must still choose the one that's the closest fit.
    * Don't write preamble nor explanation for the answer.

    {format_instructions}

    Here are some examples:

    {examples}

    Now, let's consider:

    {example_prompt}
  """

  def prepare_context(self, action_description: str, persona: 'Persona', temp_address: str) -> Dict[str, str]:
    return with_json([
      'action_description',
      'all_game_objects',
    ])({
      'action_description': action_description,
      'all_game_objects': persona.s_mem.get_array_accessible_arena_game_objects(temp_address),
    })
  
  def postprocess(self, result: ActionGameObjectResponse):
    return result.game_object

  examples = [
    {
      'action_description_json': json.dumps('sleep in bed'),
      'all_game_objects_json': json.dumps(['bed', 'easel', 'closet', 'painting']),
      'output': ActionGameObjectResponse(game_object='bed').json(),
    },
    {
      'action_description_json': json.dumps('painting'),
      'all_game_objects_json': json.dumps(['easel', 'closet', 'sink', 'microwave']),
      'output': ActionGameObjectResponse(game_object='easel').json(),
    },
    {
      'action_description_json': json.dumps('cooking'),
      'all_game_objects_json': json.dumps(['stove', 'sink', 'fridge', 'counter']),
      'output': ActionGameObjectResponse(game_object='stove').json(),
    },
    {
      'action_description_json': json.dumps('watch TV'),
      'all_game_objects_json': json.dumps(['couch', 'TV', 'remote', 'coffee table']),
      'output': ActionGameObjectResponse(game_object='couch').json(),
    },
    {
      'action_description_json': json.dumps('study'),
      'all_game_objects_json': json.dumps(['desk', 'computer', 'chair', 'bookshelf']),
      'output': ActionGameObjectResponse(game_object='desk').json(),
    },
    {
      'action_description_json': json.dumps('talk on the phone'),
      'all_game_objects_json': json.dumps(['phone', 'charger', 'bed', 'nightstand']),
      'output': ActionGameObjectResponse(game_object='phone').json(),
    },
  ]
