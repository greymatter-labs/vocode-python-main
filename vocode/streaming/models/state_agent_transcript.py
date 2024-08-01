from typing import Any, Dict, List
from pydantic import BaseModel
import datetime

class JsonTranscript:
   version: str

class StateAgentTranscriptEntry:
   role: str
   message: str
   timestamp: str = datetime.datetime.now().isoformat()

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0" 
   entries: List[StateAgentTranscriptEntry] = []