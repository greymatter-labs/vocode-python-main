from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel
import datetime

class JsonTranscript(TypedDict):
   version: str

class StateAgentTranscriptEntry(TypedDict):
   role: str
   message: str
   timestamp: str

   def __init__(self, role: str, message: str):
      self.role = role
      self.message = message
      self.timestamp = datetime.datetime.now().isoformat()

class StateAgentTranscript(JsonTranscript):
   entries: List[StateAgentTranscriptEntry]

   def __init__(self):
      self.version = "StateAgent_v0" 
      self.entries = []