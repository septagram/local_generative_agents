from typing import List
from random import Random
from datetime import datetime
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator, conlist

from persona.common import is_valid_time, string_to_time, time_to_string
from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy

class DailyPlanItem(BaseModel):
  start: datetime = Field(description="start time with am/pm")
  end: datetime = Field(description="end time with am/pm")
  activity: str = Field(description=f"the activity persona is performing, in plain text")

  @classmethod
  def validate_time(field_name: str, value: str):
    if not is_valid_time(value):
      raise ValueError(f'Invalid {field_name} time format: "{value}". Example time format: "6:00 am".')

  @validator('start')
  def parse_start(cls, time_string: str):
    cls.validate_time(time_string)
    return string_to_time(time_string)

  @validator('end')
  def parse_end(cls, time_string: str):
    cls.validate_time(time_string)
    return string_to_time(time_string)

class DailyPlanResponse(BaseModel):
  activities: conlist(item_type=DailyPlanItem, min_items=1) = Field(description="list of activities for today")

  @validator('activities')
  def activities_do_not_intersect(cls, activities: List[DailyPlanItem]):
    """
    Validates that the activities in the daily plan do not intersect in time.

    This method checks that each activity in the plan starts after the previous one ends,
    ensuring there are no overlapping activities. The first activity's start time is considered
    the wake-up time for the persona.

    Args:
      activities: A list of DailyPlanItem objects representing the planned activities.

    Returns:
      The original list of activities if validation passes, raising a ValueError otherwise.
    """

    wake_up_time = activities[0].start
    prev: DailyPlanItem = activities[0]
    for cur in activities:
      if cur == prev:
        continue
      # For night owls, activities may continue past midnight and resume before the "wake-up" time.
      # This condition allows for time entries after midnight but before the first entry's time,
      # accommodating a schedule that doesn't strictly follow chronological order across days.
      is_past_midnight = cur.start < wake_up_time and prev.end > wake_up_time
      if cur.start < prev.end and not is_past_midnight:
        raise ValueError(f'Tasks are not in chronological order. "{prev.activity}" intersects with "{cur.activity}"')
      prev = cur
    return activities
  
  @validator('activities')
  def activities_are_continuous(cls, activities: List[DailyPlanItem]):
    rng = Random(str(activities))
    idle_activities = ["Idle", "Relax", "Rest", "Chill", "Procrastinate"]
    updated_activities: List[DailyPlanItem] = [activities[0]]

    for cur in activities:
      prev = updated_activities[-1]
      if cur == prev:
        continue

      if cur.start != prev.end:
        idle_activity = rng.choice(idle_activities)
        updated_activities.append(DailyPlanItem(start=prev.end, end=cur.start, activity=idle_activity))
      
      updated_activities.append(cur)
    
    return updated_activities


@functor
class run_gpt_prompt_daily_plan(InferenceStrategy):
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
  
  output_type = DailyPlanResponse
  config = {
    "max_tokens": 1000,
    "temperature": 1,
    "top_p": 0.8,
  }
  prompt = """
    Let's consider {firstname}:

    {commonset}

    We need to draft a daily plan for {firstname} in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).
    
    {format_instructions}
    
    The entries must be in the correct order and must not intersect. The plan starts with waking up at {wake_up_hour} and completing the morning routine, and it ends with going to sleep. What would be other items in the {firstname}'s daily plan?
  """

  def prepare_context(self, persona, wake_up_hour):
    return {
      "commonset": persona.scratch.get_str_iss(),
      "date": persona.scratch.get_str_curr_date_str(),
      "firstname": persona.scratch.get_str_firstname(),
      "wake_up_hour": datetime(1, 1, 1, hour=int(wake_up_hour), minute=int((wake_up_hour % 1) * 60), second=0, microsecond=0).strftime("%I:%M %p")
    }

  def output_parser(self):
    return PydanticOutputParser(pydantic_object=DailyPlanResponse)

  def postprocess(self, result: DailyPlanResponse):
    return [f"{time_to_string(item.start)} - {item.activity}" for item in result.activities]

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
