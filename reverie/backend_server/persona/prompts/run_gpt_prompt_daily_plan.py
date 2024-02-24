from random import Random
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate

from persona.common import is_valid_time, string_to_time
from persona.prompt_template.InferenceStrategy import JSONType, OutputType, functor, InferenceStrategy

"""
Basically the long term planning that spans a day. Returns a list of actions
that the persona will take today. Usually comes in the following form: 
'wake up and complete the morning routine at 6:00 am', 
'eat breakfast at 7:00 am',.. 
Note that the actions come without a period. 

INPUT: 
  persona: The Persona class instance 
OUTPUT: 
  a list of daily actions in broad strokes.
"""
@functor
class run_gpt_prompt_daily_plan(InferenceStrategy):
  output_type = OutputType.JSON
  config = {
    "max_tokens": 1000,
    "temperature": 1,
    "top_p": 0.8,
  }
  prompt = """
    Let's consider {firstname}:

    {commonset}

    We need to draft a daily plan for {firstname} in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm). The plan must be formatted as a single JSON array of objects, each object containing the following fields:
    
    * start: start time with am/pm
    * end: end time with am/pm
    * activity: the activity {firstname} is performing, in plain text

    The entries must be in the correct order and must not intersect. The plan starts with waking up at {wake_up_hour} and completing the morning routine, and it ends with going to sleep. What would be other items in the {firstname}'s daily plan?
  """

  def prepare_context(self, persona, wake_up_hour):
    return {
      "commonset": persona.scratch.get_str_iss(),
      "date": persona.scratch.get_str_curr_date_str(),
      "firstname": persona.scratch.get_str_firstname(),
      "wake_up_hour": f"{str(wake_up_hour)}:00 am"
    }

  def validate_json(self, json: JSONType):
    if not isinstance(json, list):
      return "Invalid JSON format (expected a JSON array)"
    if not all(isinstance(item, dict) and 'start' in item and 'end' in item and 'activity' in item for item in json):
      return "Invalid JSON format (expected an array of objects with 'start', 'end' and 'activity' fields)"
    wake_up_time = string_to_time(json[0]["start"])
    prev_time = None
    prev_task = None
    for item in json:
      for field in ["start", "end"]:
        if not is_valid_time(item[field]):
          return f'Invalid {field} time format: "{item[field]}". Example time format: "6:00 am".'
      time = string_to_time(item["start"])
      # For night owls, activities may continue past midnight and resume before the "wake-up" time.
      # This condition allows for time entries after midnight but before the first entry's time,
      # accommodating a schedule that doesn't strictly follow chronological order across days.
      is_past_midnight = time < wake_up_time and prev_time > wake_up_time
      if prev_time and time < prev_time and not is_past_midnight:
        return f'Tasks are not in chronological order. "{prev_task}" intersects with "{item["activity"]}"'
      prev_time = string_to_time(item["end"])
      prev_task = item["activity"]

  def extract_json(self, json: JSONType):
    rng = Random(str(json))
    activities = ["Relax", "Rest", "Chill", "Procrastinate"]
    result = []
    for i, item in enumerate(json):
      if i != 0:
        start = item['start']
        prev_end = json[i-1]['end']
        if string_to_time(start) != string_to_time(prev_end):
          random_activity = rng.choice(activities)
          result.append(f"{prev_end} - {random_activity}")
      result.append(f"{item['start']} - {item['activity']}")
    return result
    # return [line for line in output.split('\n') if line.strip() and line[0].isdigit()]

  def fallback(self, persona, wake_up_hour):
    return [
      '6:00 am - wake up and complete the morning routine', 
      '7:00 am - eat breakfast', 
      '8:00 am - read a book', 
      '12:00 pm - have lunch', 
      '1:00 pm - take a nap', 
      '4:00 pm - relax', 
      '7:00 pm - watch TV', 
      '8:00 pm - relax', 
      '11:00 pm - go to bed',
    ]
