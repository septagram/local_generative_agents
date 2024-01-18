class HourlyScheduleItem:
  def __init__(self, task: str, start_time: int, duration: int = None):
    self.task = task
    self.start_time = start_time
    self.duration = duration

  def __repr__(self):
    return f"HourlyScheduleItem(task={self.task}, start_time={self.start_time}, duration={self.duration})"
