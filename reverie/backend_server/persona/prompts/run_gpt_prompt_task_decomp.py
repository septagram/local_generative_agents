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

import datetime
from typing import List, Optional
from persona.prompt_template.ResponseModel import ResponseModel, BaseModel, Field, validator, conlist

from persona.common import HourlyScheduleItem
from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy

class TaskDecompSubtask(BaseModel):
  i: int = Field(description='index, sequential number representing the order of the subtask')
  action: str = Field(..., description='brief description of the subtask being performe')
  duration: int = Field(..., description='time allocated for this subtask, in minutes', min=1)
  timeLeft: Optional[int] = Field(description="remaining time until the activity's completion, in minutes", min=0)

class TaskDecompResponse(ResponseModel):
  subtasks: conlist(item_type=TaskDecompSubtask, min_items=1) = Field(..., description='list of subtasks')

  @validator('subtasks')
  def evenly_split_duration(cls, subtasks: List[TaskDecompSubtask], values) -> Optional[List[TaskDecompSubtask]]:
    if values['context']:
      total_duration = sum(subtask.duration for subtask in subtasks)
      expected_duration = values['context']['duration']
      if total_duration != expected_duration:
        adjustment_ratio = expected_duration / total_duration
        return [
          TaskDecompSubtask(
            i=subtask.i,
            action=subtask.action,
            duration=int(round(subtask.duration * adjustment_ratio))
          )
          for subtask in subtasks
        ]
    return subtasks

"""
A few shot decomposition of a task given the task description.

Persona state: identity stable set, curr_date_str, first_name.

INPUT: 
  persona: The Persona class instance.
  task: the description of the task at hand in str form
        (e.g., "waking up and starting her morning routine").
  duration: an integer that indicates the number of minutes this task is 
            meant to last (e.g., 60).
OUTPUT: 
  a list of list where the inner list contains the decomposed task 
  description and the number of minutes the task is supposed to last. 
EXAMPLE OUTPUT: 
  [['going to the bathroom', 5], ['getting dressed', 5], 
    ['eating breakfast', 15], ['checking her email', 5], 
    ['getting her supplies ready for the day', 15], 
    ['starting to work on her painting', 15]] 
"""
@functor
class run_gpt_prompt_task_decomp(InferenceStrategy):
  output_type = TaskDecompResponse
  config = {
    "max_tokens": 1000,
    "temperature": 0.5,
    "top_p": 1,
  }
  prompt = """
    Let's perform task decomposition, breaking down a larger activity into smaller, manageable subtasks.

    {format_instructions}

    Here's an example of how it can be done:

    Name: Kelly Bronson
    Age: 35
    Backstory: Kelly always wanted to be a teacher, and now she teaches kindergarten. During the week, she dedicates herself to her students, but on the weekends, she likes to try out new restaurants and hang out with friends. She is very warm and friendly, and loves caring for others.
    Personality: sweet, gentle, meticulous
    Location: Kelly is in an older condo that has the following areas: {{kitchen, bedroom, dining, porch, office, bathroom, living room, hallway}}.
    Currently: Kelly is a teacher during the school year. She teaches at the school but works on lesson plans at home. She is currently living alone in a single bedroom condo.
    Daily plan requirement: Kelly is planning to teach during the morning and work from home in the afternoon.

    Today is Saturday May 10. From 08:00am ~ 09:00am, Kelly is planning on having breakfast, from 09:00am ~ 12:00pm, Kelly is planning on working on the next day's kindergarten lesson plan, and from 12:00 ~ 13pm, Kelly is planning on taking a break. 

    Given the total duration of 180 minutes for this task, here's how Kelly's subtasks can be represented in a JSON array:

    [
      {{"i": 1, "action": "Reviewing curriculum standards", "duration": 15, "timeLeft": 165}},
      {{"i": 2, "action": "Brainstorming lesson ideas", "duration": 30, "timeLeft": 135}},
      {{"i": 3, "action": "Creating the lesson plan", "duration": 30, "timeLeft": 105}},
      {{"i": 4, "action": "Creating materials for the lesson", "duration": 30, "timeLeft": 75}},
      {{"i": 5, "action": "Taking a short break", "duration": 15, "timeLeft": 60}},
      {{"i": 6, "action": "Reviewing the lesson plan", "duration": 30, "timeLeft": 30}},
      {{"i": 7, "action": "Making final adjustments to the lesson plan", "duration": 15, "timeLeft": 15}},
      {{"i": 8, "action": "Printing the lesson plan", "duration": 10, "timeLeft": 5}},
      {{"i": 9, "action": "Packing the lesson plan in her bag", "duration": 5, "timeLeft": 0}}
    ]

    Now, let's consider {firstname}, who is about to perform the task "{task}".

    {commonset}
    {surrounding_schedule}

    In 5 min increments, list the subtasks {firstname} does when performing the task "{task}" from {time_range} (total duration in minutes {duration}) as a JSON array in the format specified.
  """

  def prepare_context(self, persona, schedule_item: HourlyScheduleItem):
    # The complex part is producing the surrounding schedule.
    # Here's an example:
    #
    # Today is Saturday June 25. From 00:00 ~ 06:00am, Maeve is 
    # planning on sleeping, 06:00 ~ 07:00am, Maeve is 
    # planning on waking up and doing her morning routine, 
    # and from 07:00am ~08:00am, Maeve is planning on having 
    # breakfast.  

    self.schedule_item = schedule_item
    firstname = persona.scratch.get_str_firstname()
    schedule = persona.scratch.f_daily_schedule_hourly_org
    schedule_item_index = schedule.index(schedule_item)
    start_index = max(schedule_item_index - 1, 0)
    end_index = min(schedule_item_index + 2, len(schedule) - 1)
    surrounding_schedule_items = schedule[start_index:end_index]

    primary_time_range = None
    summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
    summ_str += f'From '
    for cur_item in surrounding_schedule_items: 
      start_time_str = self.minutes_to_time_string(cur_item.start_time)
      end_time_str = self.minutes_to_time_string(cur_item.start_time + cur_item.duration)
      cur_time_range = f'{start_time_str} ~ {end_time_str}'
      summ_str += f'{cur_time_range}, {firstname} is planning on "{cur_item.task}", '
      if cur_item is schedule_item:
        primary_time_range = f'{start_time_str} ~ {end_time_str}'
    summ_str = summ_str[:-2] + "."

    return {
      "commonset": persona.scratch.get_str_iss(),
      "surrounding_schedule": summ_str,
      "firstname": firstname,
      "task": schedule_item.task,
      "time_range": primary_time_range,
      "duration": schedule_item.duration
    }

  def postprocess(self, response: TaskDecompResponse):
    return [[subtask.action, subtask.duration] for subtask in response.subtasks]

  def fallback(self, persona, schedule_item):
    return [[schedule_item.task, schedule_item.duration]]

  def minutes_to_time_string(self, minutes):
    time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") 
            + datetime.timedelta(minutes=minutes)) 
    return time.strftime("%H:%M %p").lower()
