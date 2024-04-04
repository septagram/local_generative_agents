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

from persona.common import is_valid_time
from persona.prompt_template.InferenceStrategy import JSONType, OutputType, functor, InferenceStrategy

@functor
class run_gpt_prompt_wake_up_hour(InferenceStrategy):
  output_type = OutputType.JSON
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

  def validate_json(self, json: JSONType):
    if "time" not in json:
      return "Missing time value"
    if not is_valid_time(json['time'], require_am_pm=True) and not is_valid_time(json['time'], require_am_pm=False):
      return "Invalid time format"

  def extract_json(self, json: JSONType):
    return re.search(r"^\s*([012]?\d:\d\d)\b", json['time']).group(1)

  def fallback(self, persona):
    return "08:00"
