import datetime
import json
import re
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, root_validator

from vocode.streaming.models.memory_dependency import MemoryDependency


class StateAgentTranscriptRole(str, Enum):
    BOT = "message.bot"
    HUMAN = "human"
    ACTION_FINISH = "action-finish"
    DEBUG = "debug"


class StateAgentDebugMessageType(str, Enum):
    ACTION_INOKE = "action_invoke"
    ACTION_ERROR = "action_error"
    HANDLE_STATE = "handle_state"
    BRANCH_DECISION = "branch_decision"
    # for errors that indicate a bug in our code
    INVARIANT_VIOLATION = "invariant_violation"


class JsonTranscript(BaseModel):
    version: str

    class Config:
        use_enum_values = True


class StateAgentTranscriptEntry(BaseModel):
    role: StateAgentTranscriptRole
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

    class Config:
        use_enum_values = True


class StateAgentTranscriptMessage(StateAgentTranscriptEntry):
    role: StateAgentTranscriptRole
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

    class Config:
        use_enum_values = True


class StateAgentTranscriptDebugEntry(StateAgentTranscriptEntry):
    role: StateAgentTranscriptRole = StateAgentTranscriptRole.DEBUG
    type: StateAgentDebugMessageType


class StateAgentTranscriptActionFinish(StateAgentTranscriptEntry):
    role: StateAgentTranscriptRole = StateAgentTranscriptRole.ACTION_FINISH
    action_name: str
    runtime_inputs: dict

    def __init__(self, **data):
        super().__init__(**data)
        self.runtime_inputs = self.scrub_secrets(self.runtime_inputs)

    @staticmethod
    def scrub_secrets(runtime_inputs: dict) -> dict:
        secret_patterns = [
            r"(?i)pass(word)?",  # Matches password, pass
            r"(?i)secret",
            r"(?i)(api[_-]?key|app[_-]?key)",  # Matches api_key, apikey, app_key, appkey
            r"(?i)token",
            r"(?i)(auth|access|refresh)[_-]?token",  # Matches auth_token, access_token, refresh_token
            r"(?i)private[_-]?key",
            r"(?i)session[_-]?id",
            r"(?i)(oauth|auth)[_-]?(token|key)",  # Matches oauth_token, auth_key, etc.
            r"(?i)jwt",  # JSON Web Token
            r"(?i)(encryption|cipher)[_-]?key",
            r"(?i)client[_-]?(id|secret)",  # Matches client_id, client_secret
            r"(?i)(db|database)[_-]?(pass|password|secret)",  # Matches db_pass, database_password, etc.
            r"(?i)(aws|amazon)[_-]?(secret|key|token)",  # Matches aws_secret, amazon_key, etc.
            r"(?i)(github|gitlab|bitbucket)[_-]?(token|key|secret)",  # For version control platforms
            r"(?i)(stripe|paypal|braintree)[_-]?(key|token|secret)",  # For payment gateways
            r"(?i)(ssh|sftp)[_-]?(key|password)",
            r"(?i)certificate[_-]?(key|password)",
            r"(?i)(encryption|cipher)[_-]?(key|password)",
            r"(?i)(auth|authentication)[_-]?(key|token|secret|password)",
            r"(?i)(access|secret)[_-]?(key|token)",
            r"(?i)([a-z0-9_-]+\.)?credential",  # Matches any word ending with .credential or just credential
        ]

        def scrub_dict(d):
            scrubbed = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    scrubbed[k] = scrub_dict(v)
                elif any(re.search(pattern, k) for pattern in secret_patterns):
                    scrubbed[k] = "*****"
                else:
                    scrubbed[k] = v
            return scrubbed

        return scrub_dict(runtime_inputs)


class StateAgentTranscriptActionInvoke(StateAgentTranscriptDebugEntry):
    type: StateAgentDebugMessageType = StateAgentDebugMessageType.ACTION_INOKE
    message: str = "action invoked"
    state_id: str
    action_name: str


class StateAgentTranscriptBranchDecision(StateAgentTranscriptDebugEntry):
    type: StateAgentDebugMessageType = StateAgentDebugMessageType.BRANCH_DECISION
    message: str = "branch decision"
    ai_prompt: str
    ai_tool: dict[str, str]
    ai_response: str
    internal_edges: List[dict]
    original_state: dict


class StateAgentTranscriptActionError(StateAgentTranscriptDebugEntry):
    type: StateAgentDebugMessageType = StateAgentDebugMessageType.ACTION_ERROR
    action_name: str
    raw_error_message: str


class StateAgentTranscriptHandleState(StateAgentTranscriptDebugEntry):
    type: StateAgentDebugMessageType = StateAgentDebugMessageType.HANDLE_STATE
    message: str = ""
    state_id: str
    generated_label: str
    memory_dependencies: Optional[List[MemoryDependency]]
    memory_values: dict
    trigger: Optional[str] = (
        None  # call trace: used only for logging switch behavior of the background task, for now
    )


class StateAgentTranscriptInvariantViolation(StateAgentTranscriptDebugEntry):
    type: StateAgentDebugMessageType = StateAgentDebugMessageType.INVARIANT_VIOLATION
    original_state: Optional[dict] = None
    extra_info: Optional[dict] = None


TranscriptEntryType = Union[
    StateAgentTranscriptEntry,
    StateAgentTranscriptActionInvoke,
    StateAgentTranscriptActionError,
    StateAgentTranscriptHandleState,
    StateAgentTranscriptInvariantViolation,
]


class StateAgentTranscript(JsonTranscript):
    version: str = "StateAgent_v0"
    entries: List[TranscriptEntryType] = []

    @root_validator(pre=True)
    def parse_entries(cls, values):
        if "entries" in values:
            parsed_entries = []
            for entry in values["entries"]:
                if entry["role"] == StateAgentTranscriptRole.ACTION_FINISH:
                    parsed_entries.append(StateAgentTranscriptActionFinish(**entry))
                elif "type" in entry:
                    if entry["type"] == StateAgentDebugMessageType.ACTION_INOKE:
                        parsed_entries.append(StateAgentTranscriptActionInvoke(**entry))
                    elif entry["type"] == StateAgentDebugMessageType.ACTION_ERROR:
                        parsed_entries.append(StateAgentTranscriptActionError(**entry))
                    elif entry["type"] == StateAgentDebugMessageType.HANDLE_STATE:
                        parsed_entries.append(StateAgentTranscriptHandleState(**entry))
                    elif entry["type"] == StateAgentDebugMessageType.BRANCH_DECISION:
                        parsed_entries.append(
                            StateAgentTranscriptBranchDecision(**entry)
                        )
                    elif (
                        entry["type"] == StateAgentDebugMessageType.INVARIANT_VIOLATION
                    ):
                        parsed_entries.append(
                            StateAgentTranscriptInvariantViolation(**entry)
                        )
                else:
                    parsed_entries.append(StateAgentTranscriptMessage(**entry))
            values["entries"] = parsed_entries
        return values

    class Config:
        use_enum_values = True
