from typing import Any, Dict, List
from pydantic import BaseModel
import datetime

class JsonTranscript(BaseModel):
   version: str

class StateAgentTranscriptEntry(BaseModel):
   role: str
   message: str
   timestamp: str

   def __init__(self, role, message):
      self.timestamp = datetime.datetime.now().isoformat()
      self.role = role
      self.message = message

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []