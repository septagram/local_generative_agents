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

from persona.prompt_template.ResponseModel import ResponseModel, Field, validator
from typing import Any, Dict, Optional

from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy

class ActObjDescResponse(ResponseModel):
  object: str = Field(..., description='object name')
  user: str = Field(..., description="user's first name")
  state: str = Field(..., description="description of the state the object is in")

  @validator('object')
  def object_name_match(cls, object: str, values: Dict[str, Any]):
    if values['context']:
      expected_object = values['context']['object_name']
      if object.lower() != expected_object.lower():
        raise ValueError(f"Object name mismatch: {expected_object} expected")
    return object

  @validator('user')
  def user_name_match(cls, user: str, values: Dict[str, Any]):
    if values['context']:
      expected_name = values['context']['firstname']
      if user.lower() != expected_name.lower():
        raise ValueError(f"User name mismatch: {expected_name} expected")
    return user

@functor
class run_gpt_prompt_act_obj_desc(InferenceStrategy):
  output_type = ActObjDescResponse
  config = {
    "max_tokens": 50,
    "temperature": 0,
    "top_p": 1,
  }
  prompt = """
    We want to write an object description and to understand the state of an object that is being used by someone.
    
    {format_instructions}

    For example, if Jack is fixing the generator, the description would state:

    {{"object":"generator","user":"Jack","state":"being fixed"}}

    Now, let's consider {object_name}. {firstname} is currently performing the task "{action_description}", interacting with the {object_name}. Describe the interaction in the same form as above.
  """

  def prepare_context(self, act_game_object: str, act_desp: str, persona) -> Dict[str, str]:
    return {
      "object_name": act_game_object,
      "action_description": act_desp,
      "firstname": persona.scratch.get_str_firstname(),
    }
  
  def postprocess(self, response: ActObjDescResponse) -> str:
    return response.state
  
  def fallback(self, act_game_object: str, act_desp: str, persona) -> str:
    return f'being used by {persona.scratch.get_str_firstname()}'
