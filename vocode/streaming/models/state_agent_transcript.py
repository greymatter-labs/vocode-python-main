from typing import Any, Dict, List
from pydantic import BaseModel

class JsonTranscript(BaseModel):
   version: str

class StateAgentTranscriptEntry(BaseModel):
   role: str
   message: str

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []