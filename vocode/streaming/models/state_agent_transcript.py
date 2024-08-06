from typing import Any, Dict, List, TypedDict
from pydantic import BaseModel, Field
import datetime

class JsonTranscript(BaseModel):
   version: str

class StateAgentTranscriptEntry(BaseModel):
   role: str # message.bot, human, action-finish, or debug
   message: str
   timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class StateAgentTranscriptDebugEntry(StateAgentTranscriptEntry):
   role: str = "debug"
   type: str # action_invoke, action_error, handle_state, or invariant_violation

class StateAgentTranscriptActionInvoke(StateAgentTranscriptDebugEntry):
   type: str = "action_invoke"
   state_id: str
   action_name: str

class StateAgentTranscriptActionError(StateAgentTranscriptDebugEntry):
   type: str = "action_error"
   action_name: str
   debug_message: str
   raw_error_message: str

class StateAgentTranscriptHandleState(StateAgentTranscriptDebugEntry):
   type: str = "handle_state"
   state_id: str

class StateAgentTranscriptInvariantViolation(StateAgentTranscriptDebugEntry):
   type: str = "invariant_violation"
   message: str

class StateAgentTranscript(JsonTranscript):
   version: str = "StateAgent_v0"
   entries: List[StateAgentTranscriptEntry] = []