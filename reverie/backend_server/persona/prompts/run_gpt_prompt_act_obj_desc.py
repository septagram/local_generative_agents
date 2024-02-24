from typing import Dict, Optional

from persona.prompt_template.InferenceStrategy import JSONType, OutputType, functor, InferenceStrategy

@functor
class run_gpt_prompt_act_obj_desc(InferenceStrategy):
  output_type = OutputType.JSON
  config = {
    "max_tokens": 50,
    "temperature": 0,
    "top_p": 1,
  }
  prompt = """
    We want to write an object description and to understand the state of an object that is being used by someone. For example, if Jack is fixing the generator, the description would state:

    {{"object":"generator","user":"Jack","state":"being fixed"}}

    Now, let's consider {object_name}. {firstname} is currently performing the task "{action_description}", interacting with the {object_name}. Describe the interaction in the same form as above.
  """

  def prepare_context(self, act_game_object: str, act_desp: str, persona) -> Dict[str, str]:
    return {
      "object_name": act_game_object,
      "action_description": act_desp,
      "firstname": persona.scratch.get_str_firstname(),
    }
  
  def validate_json(self, json: JSONType) -> Optional[str]:
    # Check for the required fields in the JSON object
    required_fields = ["object", "user", "state"]
    for field in required_fields:
      if field not in json:
        return f"Missing field: {field}"
    # Check if the "object" field matches the lowercased object_name property
    if json["object"].lower() != self.context_variables['object_name'].lower():
      return "Object name mismatch"
    # Check if the "object" field matches the lowercased object_name property
    if json["user"] != self.context_variables['firstname']:
      return "Object name mismatch"
  
  def extract_json(self, json: JSONType) -> str:
    return json['state']
  
  def fallback(self, act_game_object: str, act_desp: str, persona) -> str:
    return f'being used by {persona.scratch.get_str_firstname()}'
