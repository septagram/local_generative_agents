import json
from collections import namedtuple

from utils import debug_cache_read, debug_cache_write

Message = namedtuple('Message', ['content'])
Choice = namedtuple('Choice', ['finish_reason', 'message'])

class CachedValue:
  def __init__(self, content: str):
    message = Message(content)
    choice = Choice("stop", message)
    self.choices = [choice]

  def __str__(self):
    return self.choices[0].message.content

class DebugCache:
  _instance = None
  cache = {}

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(DebugCache, cls).__new__(cls)
      try:
        with open('debug_cache.json', 'r') as f:
          cls.cache = json.load(f)
      except (FileNotFoundError, json.JSONDecodeError):
        cls.cache = {}
    return cls._instance

  @classmethod
  async def read(cls, prompt: str):
    if debug_cache_read and prompt in cls._instance.cache:
      return CachedValue(cls._instance.cache[prompt])

  @classmethod
  def write(cls, prompt: str, response: any):
    if debug_cache_write:
      cls._instance.cache[prompt] = str(response.choices[0].message.content)
      with open('debug_cache.json', 'w') as f:
        json.dump(cls._instance.cache, f)

DebugCache()
