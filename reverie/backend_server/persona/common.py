import re
import datetime

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
