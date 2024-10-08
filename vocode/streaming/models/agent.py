from enum import Enum
from typing import List, Optional, Union

from langchain.prompts import PromptTemplate
from pydantic import validator
from vocode.streaming.models.actions import ActionConfig, FunctionCall
from vocode.streaming.models.call_type import CallType
from vocode.streaming.models.message import BaseMessage

from .model import BaseModel, TypedModel
from .vector_db import VectorDBConfig

FILLER_AUDIO_DEFAULT_SILENCE_THRESHOLD_SECONDS = 0.5
LLM_AGENT_DEFAULT_TEMPERATURE = 0.5
LLM_AGENT_DEFAULT_MAX_TOKENS = 256
LLM_AGENT_DEFAULT_MODEL_NAME = "text-curie-001"
# CHAT_GPT_AGENT_DEFAULT_MODEL_NAME = "gpt-4-1106-preview"
# ACTION_AGENT_DEFAULT_MODEL_NAME = "gpt-4-1106-preview"
CHAT_GPT_AGENT_DEFAULT_MODEL_NAME = "accounts/fireworks/models/llama-v3-70b-instruct"
ACTION_AGENT_DEFAULT_MODEL_NAME = "Qwen/Qwen1.5-72B-Chat-GPTQ-Int4"
MISTRAL_AGENT_DEFAULT_MODEL_NAME = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
MISTRAL_ACTION_AGENT_DEFAULT_MODEL_NAME = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
CHAT_ANTHROPIC_DEFAULT_MODEL_NAME = "claude-v1"
CHAT_VERTEX_AI_DEFAULT_MODEL_NAME = "chat-bison@001"
AZURE_OPENAI_DEFAULT_API_TYPE = "azure"
AZURE_OPENAI_DEFAULT_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_DEFAULT_ENGINE = "gpt-35-turbo"


class AgentType(str, Enum):
    BASE = "agent_base"
    LLM = "agent_llm"
    CHAT_GPT_ALPHA = "agent_chat_gpt_alpha"
    CHAT_GPT = "agent_chat_gpt"
    CHAT_ANTHROPIC = "agent_chat_anthropic"
    CHAT_VERTEX_AI = "agent_chat_vertex_ai"
    ECHO = "agent_echo"
    GPT4ALL = "agent_gpt4all"
    LLAMACPP = "agent_llamacpp"
    INFORMATION_RETRIEVAL = "agent_information_retrieval"
    RESTFUL_USER_IMPLEMENTED = "agent_restful_user_implemented"
    WEBSOCKET_USER_IMPLEMENTED = "agent_websocket_user_implemented"
    ACTION = "agent_action"
    MISTRAL = "agent_mistral"
    COMMAND = "agent_command"


class FillerAudioConfig(BaseModel):
    silence_threshold_seconds: float = FILLER_AUDIO_DEFAULT_SILENCE_THRESHOLD_SECONDS
    use_phrases: bool = True
    use_typing_noise: bool = False

    @validator("use_typing_noise")
    def typing_noise_excludes_phrases(cls, v, values):
        if v and values.get("use_phrases"):
            values["use_phrases"] = False
        if not v and not values.get("use_phrases"):
            raise ValueError("must use either typing noise or phrases for filler audio")
        return v


class WebhookConfig(BaseModel):
    url: str


class AzureOpenAIConfig(BaseModel):
    api_type: str = AZURE_OPENAI_DEFAULT_API_TYPE
    api_version: Optional[str] = AZURE_OPENAI_DEFAULT_API_VERSION
    engine: str = AZURE_OPENAI_DEFAULT_ENGINE


class AgentConfig(TypedModel, type=AgentType.BASE.value):
    initial_message: Optional[BaseMessage] = None
    generate_responses: bool = True
    allowed_idle_time_seconds: Optional[float] = None
    allow_agent_to_be_cut_off: bool = True
    end_conversation_on_goodbye: bool = False
    use_filler_words: bool = False
    webhook_config: Optional[WebhookConfig] = None
    track_bot_sentiment: bool = False
    actions: Optional[List[ActionConfig]] = None
    current_call_id: Optional[int] = None
    call_type: Optional[CallType] = None
    from_phone_number: Optional[str] = None
    to_phone_number: Optional[str] = None
    transcript_analyzer_func: Optional[str] = None


class CutOffResponse(BaseModel):
    messages: List[BaseMessage] = [BaseMessage(text="Sorry?")]


class LLMAgentConfig(AgentConfig, type=AgentType.LLM.value):
    prompt_preamble: str
    expected_first_prompt: Optional[str] = None
    model_name: str = LLM_AGENT_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    cut_off_response: Optional[CutOffResponse] = None


class ChatGPTAgentConfig(AgentConfig, type=AgentType.CHAT_GPT.value):
    prompt_preamble: str
    expected_first_prompt: Optional[str] = None
    model_name: str = CHAT_GPT_AGENT_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    cut_off_response: Optional[CutOffResponse] = None
    azure_params: Optional[AzureOpenAIConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None
    pending_action: Optional[FunctionCall] = None
    use_filler_words: bool = False
    language: Optional[str] = "en-US"


class CommandAgentConfig(AgentConfig, type=AgentType.COMMAND.value):
    prompt_preamble: Optional[str] = None
    user_json_prompt: Optional[dict] = {}
    expected_first_prompt: Optional[str] = None
    model_name: str = CHAT_GPT_AGENT_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    cut_off_response: Optional[CutOffResponse] = None
    azure_params: Optional[AzureOpenAIConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None
    pending_action: Optional[FunctionCall] = None
    use_filler_words: bool = False
    use_streaming: bool = False
    allow_interruptions: bool = False
    language: Optional[str] = "en-US"
    ai_profile_id: Optional[str] = None
    timezone: Optional[str] = "America/Los_Angeles"


class MistralAgentConfig(AgentConfig, type=AgentType.MISTRAL.value):
    prompt_preamble: str
    expected_first_prompt: Optional[str] = None
    model_name: str = MISTRAL_AGENT_DEFAULT_MODEL_NAME
    temperature: float = LLM_AGENT_DEFAULT_TEMPERATURE
    max_tokens: int = LLM_AGENT_DEFAULT_MAX_TOKENS
    cut_off_response: Optional[CutOffResponse] = None
    azure_params: Optional[AzureOpenAIConfig] = None
    vector_db_config: Optional[VectorDBConfig] = None


class ChatAnthropicAgentConfig(AgentConfig, type=AgentType.CHAT_ANTHROPIC.value):
    prompt_preamble: str
    model_name: str = CHAT_ANTHROPIC_DEFAULT_MODEL_NAME
    max_tokens_to_sample: int = 200


class ChatVertexAIAgentConfig(AgentConfig, type=AgentType.CHAT_VERTEX_AI.value):
    prompt_preamble: str
    model_name: str = CHAT_VERTEX_AI_DEFAULT_MODEL_NAME
    generate_responses: bool = False  # Google Vertex AI doesn't support streaming


class LlamacppAgentConfig(AgentConfig, type=AgentType.LLAMACPP.value):
    prompt_preamble: str
    llamacpp_kwargs: dict = {}
    prompt_template: Optional[Union[PromptTemplate, str]] = None


class InformationRetrievalAgentConfig(
    AgentConfig, type=AgentType.INFORMATION_RETRIEVAL.value
):
    recipient_descriptor: str
    caller_descriptor: str
    goal_description: str
    fields: List[str]
    # TODO: add fields for IVR, voicemail


class EchoAgentConfig(AgentConfig, type=AgentType.ECHO.value):
    pass


class GPT4AllAgentConfig(AgentConfig, type=AgentType.GPT4ALL.value):
    prompt_preamble: str
    model_path: str
    generate_responses: bool = False


class RESTfulUserImplementedAgentConfig(
    AgentConfig, type=AgentType.RESTFUL_USER_IMPLEMENTED.value
):
    class EndpointConfig(BaseModel):
        url: str
        method: str = "POST"

    respond: EndpointConfig
    generate_responses: bool = False
    # generate_response: Optional[EndpointConfig]


class RESTfulAgentInput(BaseModel):
    conversation_id: str
    human_input: str


class RESTfulAgentOutputType(str, Enum):
    BASE = "restful_agent_base"
    TEXT = "restful_agent_text"
    END = "restful_agent_end"


class RESTfulAgentOutput(TypedModel, type=RESTfulAgentOutputType.BASE):
    pass


class RESTfulAgentText(RESTfulAgentOutput, type=RESTfulAgentOutputType.TEXT):
    response: str


class RESTfulAgentEnd(RESTfulAgentOutput, type=RESTfulAgentOutputType.END):
    pass
