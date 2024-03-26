import json
import re
import datetime
from typing import Any, Dict, List

class HourlyScheduleItem:
  def __init__(self, task: str, start_time: int, duration: int = None):
    self.task = task
    self.start_time = start_time
    self.duration = duration

  def __repr__(self):
    return f"HourlyScheduleItem(task={self.task}, start_time={self.start_time}, duration={self.duration})"

def is_valid_time(time_string: str, require_am_pm=True):
  regexp_ampm = r"^\s*[012]?\d:\d\d\b\s+[ap]m$"
  regexp_24 = r"^\s*[012]?\d:\d\d$"
  return bool(re.match(regexp_ampm if require_am_pm else regexp_24, time_string, re.IGNORECASE))

def string_to_time(time_string: str, require_am_pm=True) -> datetime.time:
  time_format = "%I:%M %p" if require_am_pm else "%H:%M"
  return datetime.datetime.strptime(time_string.strip().upper(), time_format).time()

def time_to_string(time: datetime.time, include_am_pm=True) -> str:
  time_format = "%I:%M %p" if include_am_pm else "%H:%M"
  return time.strftime(time_format).lower()

def with_transformation_suffix(keys: List[str], suffix: str, transformer: lambda value: Any):
  def curried_transformer(dict: Dict[str, Any]) -> Dict[str, Any]:
    transformed_dict = dict.copy()
    for key in keys:
      if key not in dict:
        raise Exception(f"Key {key} not found in dictionary")
      transformed_value = transformer(dict[key])
      transformed_dict[f"{key}{suffix}"] = transformed_value
    return transformed_dict
  return curried_transformer

def with_json(keys: List[str]):
  return with_transformation_suffix(keys, '_json', json.dumps)

def deindent(text: str) -> str:
  # Deindent by removing the first sequence of spaces or tabs starting with \n
  # and removing all its instances
  match = re.search(r'\n[ \t]+', text)
  if match:
    first_indentation = match.group()
  else:
    first_indentation = '\n'
  return text.strip().replace(first_indentation, '\n')

if __name__ == "__main__":
  # Example usage of the functions defined in this file
  for time_string in ["10:00 PM", "9:30 am"]:
    if is_valid_time(time_string):
      print(f"The time '{time_string}' is valid.")
      converted_time = string_to_time(time_string)
      print(f"Converted time: {converted_time}")
      back_to_string = time_to_string(converted_time)
      print(f"Back to string: {back_to_string}")
    else:
      print(f"The time '{time_string}' is not valid.")
