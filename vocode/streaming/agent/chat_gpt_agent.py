import asyncio
import json
import logging

from typing import Any, Dict, List, Optional, Tuple, Union
import aiohttp

import openai
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Optional, Tuple

import logging
from pydantic import BaseModel

from vocode import getenv
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.agent.utils import (
    format_openai_chat_messages_from_transcript,
    collate_response_async,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.vector_db.factory import VectorDBFactory

from telephony_app.models.call_type import CallType
from telephony_app.utils.call_information_handler import update_call_transcripts, get_company_primary_phone_number, \
    get_telephony_id_from_internal_id
from telephony_app.utils.transfer_call_handler import transfer_call
from telephony_app.utils.twilio_call_helper import hangup_twilio_call


class ChatGPTAgent(RespondAgent[ChatGPTAgentConfig]):
    def __init__(
            self,
            agent_config: ChatGPTAgentConfig,
            action_factory: ActionFactory = ActionFactory(),
            logger: Optional[logging.Logger] = None,
            openai_api_key: Optional[str] = None,
            vector_db_factory=VectorDBFactory(),
    ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )

        if agent_config.azure_params:
            self.aclient = AsyncAzureOpenAI(api_version=agent_config.azure_params.api_version,
                                            api_key=getenv("AZURE_OPENAI_API_KEY"),
                                            azure_endpoint=getenv("AZURE_OPENAI_API_BASE"))

            self.client = AzureOpenAI(api_version=agent_config.azure_params.api_version,
                                      api_key=getenv("AZURE_OPENAI_API_KEY"),
                                      azure_endpoint=getenv("AZURE_OPENAI_API_BASE"))
        else:
            # mistral configs
            self.aclient = AsyncOpenAI(api_key="EMPTY", base_url=getenv("MISTRAL_API_BASE"))
            self.client = OpenAI(api_key="EMPTY", base_url=getenv("MISTRAL_API_BASE"))
            self.fclient = AsyncOpenAI(api_key="functionary", base_url=getenv("FUNCTIONARY_API_BASE"))

            # openai.api_type = "open_ai"
            # openai.api_version = None

            # chat gpt configs
            # openai.api_type = "open_ai"
            # openai.api_base = "https://api.openai.com/v1"
            # openai.api_version = None
            # openai.api_key = openai_api_key or getenv("OPENAI_API_KEY")

        if not self.client.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.first_response = (
            self.create_first_response(agent_config.expected_first_prompt)
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(
                self.agent_config.vector_db_config
            )

        if self.logger:
            self.logger.setLevel(logging.INFO)

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]

    def get_chat_parameters(
            self, messages: Optional[List] = None, use_functions: bool = True
    ):
        assert self.transcript is not None
        messages = messages or format_openai_chat_messages_from_transcript(
            self.transcript, self.agent_config.prompt_preamble
        )

        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            "stop": ["User:", "\n", "<|im_end|>", "?"],
        }

        if self.agent_config.azure_params is not None:
            parameters["engine"] = self.agent_config.azure_params.engine
        else:
            parameters["model"] = self.agent_config.model_name

        if use_functions and self.functions:
            parameters["functions"] = self.functions

        return parameters

    def create_first_response(self, first_prompt):
        messages = [
            (
                [{"role": "system", "content": self.agent_config.prompt_preamble}]
                if self.agent_config.prompt_preamble
                else []
            )
            + [{"role": "user", "content": first_prompt}]
        ]

        parameters = self.get_chat_parameters(messages)
        return self.client.chat.completions.create(**parameters)

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    async def run_nonblocking_checks(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message/ notification to the receiver.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message_to_pass": {
                                "type": "string",
                                "description": "The message to pass, limited to 300 characters"
                            }
                        },
                        "required": ["message_to_pass"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_call",
                    "description": "Transfer the call with a reason",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "transfer_reason": {
                                "type": "string",
                                "description": "The reason for transferring the call, limited to 120 characters"
                            }
                        },
                        "required": ["transfer_reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the call with a reason",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "end_reason": {
                                "type": "string",
                                "description": "The reason for ending the call, limited to 120 characters"
                            }
                        },
                        "required": ["end_reason"]
                    }
                }
            }
        ]
        try:
            tool_descriptions = [f"'{tool['function']['name']}': {tool['function']['description']} (Required params: {', '.join(tool['function']['parameters']['required'])})" for tool in tools]
            pretty_tool_descriptions = ', '.join(tool_descriptions)
            stringified_messages = self.transcript.to_string()
            preamble = f"""You will be provided with a conversational transcript between a caller and the receiver's assistant. During the conversation, the assistant has the following actions it can take: {pretty_tool_descriptions}.\n Your task is to infer whether, currently, the assistant is waiting for the caller to respond, or is immediately going to execute an action without waiting for a response. Return the action name from the list provided if the assistant is executing an action. If the assistant is waiting for a response, return 'None'. Return a single word."""
            system_message = {"role": "system", "content": preamble}
            transcript_message = {"role": "user", "content": stringified_messages}
            combined_messages = [system_message, transcript_message]
            chat_parameters = self.get_chat_parameters(messages=combined_messages)
            chat_parameters["temperature"] = 0
            response = await self.aclient.chat.completions.create(**chat_parameters)
            tool_classification = response.choices[0].message.content.lower().strip()
            if tool_classification in [tool["function"]["name"].lower() for tool in tools]:
                self.logger.info(f"Tool classification: {tool_classification}")
                chat = format_openai_chat_messages_from_transcript(self.transcript)
                toolResponse = await self.fclient.chat.completions.create(
                    model="meetkai/functionary-small-v2.2",
                    messages=chat,
                    tools=tools,
                    tool_choice="auto"
                )
                toolResponse = toolResponse.choices[0]
                message_content = toolResponse.message.content
                tool_call = toolResponse.message.tool_calls[0]
                if tool_call:
                    yield FunctionCall({"name": tool_call.function.name, "arguments": tool_call.function.arguments})
            else:
                self.logger.info(f"Tool classification: None")
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        return
    
    async def respond(
            self,
            human_input,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        assert self.transcript is not None
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            text = self.first_response
        else:
            chat_parameters = self.get_chat_parameters()
            chat_completion = await self.aclient.chat.completions.create(**chat_parameters)
            text = chat_completion.choices[0].message.content
        self.logger.debug(f"LLM response: {text}")
        return text, False
    
    async def generate_response(
            self,
            human_input: str,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> AsyncGenerator[Tuple[Union[str, FunctionCall], bool], None]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield cut_off_response, False
            return
        assert self.transcript is not None

        chat_parameters = {}
        if self.agent_config.vector_db_config:
            try:
                vector_db_search_args = {
                    "query": self.transcript.get_last_user_message()[1],
                }

                has_vector_config_namespace = getattr(self.agent_config.vector_db_config, 'namespace', None)
                if has_vector_config_namespace:
                    vector_db_search_args["namespace"] = \
                        self.agent_config.vector_db_config.namespace.lower().replace(" ", "_")

                docs_with_scores = await self.vector_db.similarity_search_with_score(
                    **vector_db_search_args
                )
                docs_with_scores_str = "\n\n".join(
                    [
                        "Document: "
                        + doc[0].metadata["source"]
                        + f" (Confidence: {doc[1]})\n"
                        + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                        for doc in docs_with_scores
                    ]
                )
                vector_db_result = f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
                messages = format_openai_chat_messages_from_transcript(
                    self.transcript, self.agent_config.prompt_preamble
                )
                messages.insert(
                    -1, vector_db_result_to_openai_chat_message(vector_db_result)
                )
                chat_parameters = self.get_chat_parameters(messages)
            except Exception as e:
                self.logger.error(f"Error while hitting vector db: {e}", exc_info=True)
                chat_parameters = self.get_chat_parameters()
        else:
            chat_parameters = self.get_chat_parameters()
        chat_parameters["stream"] = True
        stream = await self.aclient.chat.completions.create(**chat_parameters)
        all_messages = []
        async for message in collate_response_async(
            openai_get_tokens(stream), get_functions=True
        ):
            all_messages.append(f"{message} ")
            yield message, True

        latest_agent_response = ''.join(all_messages)
        await self.run_nonblocking_checks()
            
        self.logger.info(f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Agent: {latest_agent_response}")

    async def transfer_call(self, telephony_id):
        if self.agent_config.call_type == CallType.INBOUND:
            company_phone_number = self.agent_config.to_phone_number
        elif self.agent_config.call_type == CallType.OUTBOUND:
            company_phone_number = self.agent_config.from_phone_number
        else:
            raise ValueError(
                f"There is no assigned call type for call {self.agent_config.current_call_id}"
            )

        phone_number_transfer_to = await get_company_primary_phone_number(
            phone_number=company_phone_number
        )

        # transfer to the primary number associated with each company; the phone number being called into is
        # associated to a singular assigned company
        await transfer_call(
            telephony_call_sid=telephony_id,
            phone_number_transfer_to=phone_number_transfer_to,
        )
