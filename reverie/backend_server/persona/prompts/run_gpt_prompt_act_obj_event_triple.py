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

from langchain_core.pydantic_v1 import BaseModel, Field

from persona.prompt_template.InferenceStrategy import ReturnType, functor, InferenceStrategy

class ActObjEventTripleResponse(BaseModel):
  object: str = Field(...)
  predicate: str = Field(...)
  interaction: str = Field(...)

@functor
class run_gpt_prompt_act_obj_event_triple(InferenceStrategy):
  output_type = ActObjEventTripleResponse
  config = {
    "max_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40, # not supported by SK
    "min_p": 0.05, # not supported by SK
  }
  example_count = 2
  example_prompt = """
    Name: {firstname}
    Action description: {action_description}
    Object: {object_name}
    Object state: {object_state}
  """
  prompt = """
    Transform natural language descriptions into structured JSON, focusing on the object, predicate, and specific status. The 'status' should reflect the primary action being performed with the object, described in a passive form, and should not include additional details unrelated to the action itself.

    {format_instructions}

    Here are some examples:

    {examples}

    Now, for a new case:

    {example_prompt}

    Based on this description, provide a single JSON object in the format shown above. The "object" field must contain object name. Do not make the "status" a generic action, such as "being used", but find a more specific word clarifying how the {object_name} is being used. In addition, exclude any extraneous details not directly related to this action. No intro nor Markdown, respond just with the JSON object.
  """

  def prepare_context(self, persona, task, act_obj_desc, object_name):
    return {
      "object_name": object_name,
      "action_description": task,
      "object_state": act_obj_desc,
      "firstname": persona.scratch.get_str_firstname(),
    }
  
  def postprocess(self, response: ActObjEventTripleResponse) -> ReturnType:
    # Check for the required fields in the JSON object
    if response.object.lower() != self.context['object_name'].lower():
      raise ValueError("Object name mismatch")
    return (response.object, response.predicate, response.interaction)
  
  def fallback(self, persona, task, act_obj_desc, object_name): 
    return (object_name, "is", "idle")

  examples = [
    {
      'firstname': 'Sam',
      'action_description': 'Sam Johnson is eating breakfast.',
      'object_name': 'table',
      'object_state': 'clear with a plate of food and a cup of coffee',
      'output': ActObjEventTripleResponse(
        object='table',
        predicate='is',
        interaction='being eaten on',
      ).json(),
    },
    {
      'firstname': 'Joon',
      'action_description': 'Joon Park is brewing coffee.',
      'object_name': 'coffee maker',
      'object_state': 'simmering',
      'output': ActObjEventTripleResponse(
        object='coffee maker',
        predicate='is',
        interaction='brewing coffee',
      ).json(),
    },
    {
      'firstname': 'Jane',
      'action_description': 'Jane Cook is sleeping.',
      'object_name': 'bed',
      'object_state': 'supported Jane during her sleep',
      'output': ActObjEventTripleResponse(
        object='bed',
        predicate='is',
        interaction='being slept in',
      ).json(),
    },
    {
      'firstname': 'Michael',
      'action_description': 'Michael Bernstein is writing email on a computer.',
      'object_name': 'computer',
      'object_state': 'in use',
      'output': ActObjEventTripleResponse(
        object='computer',
        predicate='is',
        interaction='being used to write email',
      ).json(),
    },
    {
      'firstname': 'Percy',
      'action_description': 'Percy Liang is teaching students in a classroom.',
      'object_name': 'classroom',
      'object_state': 'filled with students learning',
      'output': ActObjEventTripleResponse(
        object='classroom',
        predicate='is',
        interaction='being used for teaching',
      ).json(),
    },
    {
      'firstname': 'Merrie',
      'action_description': 'Merrie Morris is running on a treadmill.',
      'object_name': 'treadmill',
      'object_state': 'in use',
      'output': ActObjEventTripleResponse(
        object='treadmill',
        predicate='is',
        interaction='being run on',
      ).json(),
    },
  ]
