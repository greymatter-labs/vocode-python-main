from typing import Any, Dict, List
from pydantic import BaseModel
import datetime

class JsonTranscript(BaseModel):
   version: str

class StateAgentTranscriptEntry(BaseModel):
   def __init__(self, role: str, message: str):
      self.timestamp = datetime.datetime.now().isoformat()
      self.role = role
      self.message = message

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []