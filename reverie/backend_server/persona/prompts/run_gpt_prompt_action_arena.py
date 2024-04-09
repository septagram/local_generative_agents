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

import json
from typing import Any, Dict
from langchain_core.pydantic_v1 import Field, validator

from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy
from persona.prompt_template.ResponseModel import ResponseModel
from persona.common import with_json

class ActionArenaResponse(ResponseModel):
  arena: str = Field(..., description="the Arena that this character should go to")
  
  @validator('arena')
  def validate_arena(cls, arena: str, values: Dict[str, Any]):
    if values['context']:
      context = values['context']
      if arena == context['current_sector'] or arena == context['current_sector']:
        raise ValueError(f"Sector name was returned instead of an Arena name. Select one of {context['target_sector_arenas_json']}")
      if not arena in context['target_sector_arenas']:
        raise ValueError(f"Specified Arena is not in the list of available Arenas. Select one of {context['target_sector_arenas_json']}")
    return arena

@functor
class run_gpt_prompt_action_arena(InferenceStrategy):
  output_type = ActionArenaResponse
  config = {
    "temperature": 0
  }
  example_prompt = """
    {name} is in {current_arena_json} in {current_sector_json}.
    {name} is going to {target_sector_json} that has the following Arenas: {target_sector_arenas_json}
    Which Arena should {name} go to for performing the {action_description_json} Action?
  """
  prompt = """
    We need to choose an appropriate Arena for the task at hand.
    
    * Arena is a kind of location within a larger Sector. Each Sector contains one or more Arenas.
    * Stay in the current Arena if the activity can be done there. Never go into other people's rooms unless necessary.
    * Must be one of the listed Arenas verbatim. It must be an Arena and not a Sector.
    * If none of those fit very well, we must still choose the one that's the closest fit.

    {format_instructions}

    Here are some examples:

    {examples}

    Now, let's consider {name}:

    {example_prompt}
  """

  def prepare_context(self, action_description, persona, maze, act_world, act_sector):
    return with_json([
      'action_description',
      'current_sector',
      'current_arena',
      'target_sector',
      'target_sector_arenas',
    ])({
      'name': persona.scratch.get_str_name(),
      'action_description': action_description,
      'current_sector': maze.access_tile(persona.scratch.curr_tile)['sector'],
      'current_arena': maze.access_tile(persona.scratch.curr_tile)['arena'],
      'target_sector': act_sector,
      'target_sector_arenas': persona.s_mem.get_array_accessible_sector_arenas(f"{act_world}:{act_sector}"),
    })
  
  def postprocess(self, result: ActionArenaResponse):
    return result.arena

  examples = [
    {
      'name': 'Jane Anderson',
      'action_description_json': json.dumps('cooking'),
      'current_sector_json': json.dumps("Jane Anderson's house"),
      'current_arena_json': json.dumps("kitchen"),
      'target_sector_json': json.dumps("Jane Anderson's house"),
      'target_sector_arenas_json': json.dumps(['kitchen', 'bedroom', 'bathroom']),
      'output': ActionArenaResponse(arena='kitchen').json(),
    },
    {
      'name': 'Tom Watson',
      'action_description_json': json.dumps('getting coffee'),
      'current_sector_json': json.dumps("Tom Watson's apartment"),
      'current_arena_json': json.dumps("common room"),
      'target_sector_json': json.dumps("Hobbs Cafe"),
      'target_sector_arenas_json': json.dumps(['cafe']),
      'output': ActionArenaResponse(arena='cafe').json(),
    },
  ]