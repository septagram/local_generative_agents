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

def is_valid_time(time_string, require_am_pm=True):
  regexp_ampm = r"^\s*[012]?\d:\d\d\b\s+[ap]m$"
  regexp_24 = r"^\s*[012]?\d:\d\d$"
  return bool(re.match(regexp_ampm if require_am_pm else regexp_24, time_string))

def string_to_time(time_string, require_am_pm=True):
  time_format = "%I:%M %p" if require_am_pm else "%H:%M"
  return datetime.datetime.strptime(time_string, time_format)

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
