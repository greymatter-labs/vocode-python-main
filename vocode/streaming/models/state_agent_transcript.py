from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel
import datetime

class JsonTranscript(TypedDict):
   version: str

class StateAgentTranscriptEntry(TypedDict):
   role: str
   message: str
   timestamp: str = datetime.datetime.now().isoformat()

   def __init__(self, role: str, message: str):
      self.role = role
      self.message = message

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []