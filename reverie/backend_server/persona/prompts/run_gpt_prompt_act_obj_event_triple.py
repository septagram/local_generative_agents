from typing import Optional

from persona.prompt_template.InferenceStrategySK import JSONType, ReturnType, OutputType, functor, InferenceStrategySK

@functor
class run_gpt_prompt_act_obj_event_triple(InferenceStrategySK):
  output_type = OutputType.JSON
  config = {
    "max_tokens": 50,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40, # not supported by SK
    "min_p": 0.05, # not supported by SK

  }
  prompt = """
    Transform natural language descriptions into structured JSON, focusing on the object, predicate, and specific status. The 'status' should reflect the primary action being performed with the object, described in a passive form, and should not include additional details unrelated to the action itself. Here are examples:

    Name: Sam
    Action description: Sam Johnson is eating breakfast. 
    Object: table
    Object state: clear with a plate of food and a cup of coffee
    Output: {
      "object": "table",
      "predicate": "is",
      "interaction": "being eaten on"
    }
    --- 
    Name: Joon
    Action description: Joon Park is brewing coffee.
    Object: coffee maker
    Object state: simmering
    Output: {
      "object": "coffee maker",
      "predicate": "is",
      "interaction": "brewing coffee"
    }
    --- 
    Name: Jane
    Action description: Jane Cook is sleeping.
    Object: bed
    Object state: supported Jane during her sleep 
    Output: {
      "object": "bed",
      "predicate": "is",
      "interaction": "being slept in"
    }
    --- 
    Name: Michael
    Action description: Michael Bernstein is writing email on a computer. 
    Object: computer
    Object state: in use
    Output: {
      "object": "computer",
      "predicate": "is",
      "interaction": "being used to write email"
    }
    --- 
    Name: Percy
    Action description: Percy Liang is teaching students in a classroom. 
    Object: classroom
    Object state: filled with students learning
    Output: {
      "object": "classroom",
      "predicate": "is",
      "interaction": "being used for teaching"
    }
    --- 
    Name: Merrie
    Action description: Merrie Morris is running on a treadmill.
    Object: treadmill
    Object state: in use 
    Output: {
      "object": "treadmill",
      "predicate": "is",
      "interaction": "being run on"
    }

    Now, for a new case:

    Name: {{$firstname}}
    Action description: {{$action_description}}
    Object: {{$object_name}}
    Object state: {{$object_state}}

    Based on this description, provide a single JSON object in the format shown above. The "object" field must contain object name. Do not make the "status" a generic action, such as "being used", but find a more specific word clarifying how the {{$object_name}} is being used. In addition, exclude any extraneous details not directly related to this action. No intro nor Markdown, respond just with the JSON object.
  """

  def prepare_context(self, persona, task, act_obj_desc, object_name):
    return {
      "object_name": object_name,
      "action_description": task,
      "object_state": act_obj_desc,
      "firstname": persona.scratch.get_str_firstname(),
    }
  
  def validate_json(self, json: JSONType) -> Optional[str]:
    # Check for the required fields in the JSON object
    required_fields = ["object", "predicate", "interaction"]
    for field in required_fields:
      if field not in json:
        return f"Missing field: {field}"
    # Check if the "object" field matches the lowercased object_name property
    if json["object"].lower() != self.context_variables['object_name'].lower():
      return "Object name mismatch"
    
  def extract_json(self, json: JSONType) -> ReturnType:
    return (json["object"], json["predicate"], json["interaction"])
  
  def fallback(self, persona, task, act_obj_desc, object_name): 
    return (object_name, "is", "idle")
