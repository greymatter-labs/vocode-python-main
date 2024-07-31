from typing import Any, Dict, List
from pydantic import BaseModel
import datetime

class JsonTranscript(BaseModel):
   version: str

class StateAgentTranscriptEntry(BaseModel):
   role: str
   message: str
   timestamp: str

   def __init__(self, role: str, message: str):
      super().__init__(role=role, message=message, timestamp=datetime.datetime.now().isoformat())

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []