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

import re
import datetime
from typing import Union
from persona.prompt_template.ResponseModel import ResponseModel, Field, validator

from persona.common import time_to_string, validate_time
from persona.prompt_template.InferenceStrategy import functor, InferenceStrategy

class WakeUpHourResponse(ResponseModel):
  time: datetime.time = Field(description="wake up time with am/pm")

  @validator('time', pre=True)
  def parse_time(cls, time_string: Union[str, datetime.time]) -> datetime.time:
    return validate_time('wake-up', time_string, False)
  
@functor
class run_gpt_prompt_wake_up_hour(InferenceStrategy):
  output_type = WakeUpHourResponse
  config = {
    "max_tokens": 15,
    "temperature": 1,
    "top_p": 0.8,
  }
  prompt = """
    {iss}

    What will be the {lifestyle} {firstname}'s wake up time today?

    Provide the answer in a JSON format with only the actual time, without any reasoning or deliberation. Format the time in a 24-hour format like "H:mm" (H for hours, mm for minutes) and include it as the value for the "time" key in the JSON object. Do not include am/pm or any other information aside from the actual time. The answer should be in the following format: {{"time": "H:mm"}}.

    Example: 
    If the wake up time is 6:00 AM, the answer should be: {{"time": "6:00"}}
  """
  def prepare_context(self, persona):
    return {
      "iss": persona.scratch.get_str_iss(),
      "lifestyle": persona.scratch.get_str_lifestyle(),
      "firstname": persona.scratch.get_str_firstname()
    }

  def postprocess(self, response: WakeUpHourResponse):
    return time_to_string(response.time, False)

  def fallback(self, persona):
    return "08:00"
