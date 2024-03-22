import asyncio
import json
import logging
import random
import string
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)

from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Optional, Tuple
import aiohttp
import logging
from pydantic import BaseModel

from vocode import getenv
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponseMessage,
    RespondAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.models.actions import ActionInput, FunctionCall, FunctionFragment
from vocode.streaming.models.agent import CommandAgentConfig
from vocode.streaming.agent.utils import (
    format_command_function_completion_from_transcript,
    format_openai_chat_completion_from_transcript,
    format_openai_chat_messages_from_transcript,
    format_tool_completion_from_transcript,
    collate_response_async,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
    translate_message,
)
from vocode.streaming.transcriber.base_transcriber import (
    Transcription,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.action.phone_call_action import (
    TwilioPhoneCallAction,
    VonagePhoneCallAction,
)
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.vector_db.factory import VectorDBFactory

from telephony_app.models.call_type import CallType
from telephony_app.utils.call_information_handler import (
    update_call_transcripts,
    get_company_primary_phone_number,
    get_telephony_id_from_internal_id,
)
from pydantic import BaseModel, Field

from telephony_app.utils.transfer_call_handler import transfer_call
from telephony_app.utils.twilio_call_helper import hangup_twilio_call

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer 'EMPTY'",
}


class EventLog(BaseModel):
    sender: Sender
    timestamp: float = Field(default_factory=time.time)

    def to_string(self, include_timestamp: bool = False) -> str:
        raise NotImplementedError


class CommandAgent(RespondAgent[CommandAgentConfig]):
    def __init__(
        self,
        agent_config: CommandAgentConfig,
        action_factory: ActionFactory = ActionFactory(),
        logger: Optional[logging.Logger] = None,
        openai_api_key: Optional[str] = None,
        vector_db_factory=VectorDBFactory(),
    ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            "CohereForAI/c4ai-command-r-v01", trust_remote_code=False, use_fast=True
        )
        self.can_send = False
        self.conversation_id = None
        self.twilio_sid = None
        model_id = "CohereForAI/c4ai-command-r-v01"
        self.no_tool_reason = ""

        self.agent_config.pending_action = None
        if agent_config.azure_params:
            self.aclient = AsyncAzureOpenAI(
                api_version=agent_config.azure_params.api_version,
                api_key=getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=getenv("AZURE_OPENAI_API_BASE"),
            )

            self.client = AzureOpenAI(
                api_version=agent_config.azure_params.api_version,
                api_key=getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=getenv("AZURE_OPENAI_API_BASE"),
            )
        else:
            # mistral configs
            self.aclient = AsyncOpenAI(
                api_key="EMPTY", base_url="http://148.64.105.83:4000/v1"
            )
            self.client = OpenAI(
                api_key="EMPTY", base_url="http://148.64.105.83:4000/v1"
            )
            self.fclient = AsyncOpenAI(
                api_key="functionary", base_url="http://148.64.105.83:4000/v1"
            )

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
            self.logger.setLevel(logging.DEBUG)

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
            # "stop": ["User:", "\n", "<|im_end|>", "?"],
            # just ?
            "stop": ["?"],
        }

        if self.agent_config.azure_params is not None:
            parameters["engine"] = self.agent_config.azure_params.engine
        else:
            parameters["model"] = self.agent_config.model_name

        if use_functions and self.functions:
            parameters["functions"] = self.functions
        return parameters

    def get_completion_parameters(  # MARKED
        self,
        messages: Optional[List] = None,
        use_functions: bool = True,
        affirmative_phrase: Optional[str] = "",
        reason: str = "",
        did_action: str = None,
    ):
        assert self.transcript is not None
        # add an
        formatted_completion, messages = format_openai_chat_completion_from_transcript(
            self.tokenizer,
            self.transcript,
            self.agent_config.prompt_preamble,
            did_action=did_action,
            reason=reason,
        )
        # log messages
        self.logger.debug(f"Messages: {messages}")

        # self.logger.debug(f"Formatted completion: {formatted_completion}")
        parameters: Dict[str, Any] = {
            "prompt": formatted_completion,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
            # "stop": ["User:", "\n", "<|im_end|>", "?"],
            # just ?
            "stop": ["?"],
        }

        if self.agent_config.azure_params is not None:
            parameters["engine"] = self.agent_config.azure_params.engine
        else:
            parameters["model"] = self.agent_config.model_name
            # parameters["model"] = "TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ"

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

    async def check_conditions(
        self, stringified_messages: str, conditions: List[str]
    ) -> List[str]:
        true_conditions = []

        tasks = []
        for condition in conditions:
            user_message = {
                "role": "user",
                "content": stringified_messages
                + "\n\nNow, return either 'True' or 'False' depending on whether the condition: <"
                + condition.strip()
                + "> applies (True) to the conversation or not (False).",
            }

            preamble = "You will be provided a condition and a conversation. Please classify if that condition applies (True), or does not apply (False) to the provided conversation.\n\nCondition:\n"
            system_message = {"role": "system", "content": preamble + condition}
            combined_messages = [system_message, user_message]
            chat_parameters = self.get_chat_parameters(messages=combined_messages)
            task = self.aclient.chat.completions.create(**chat_parameters)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for response, condition in zip(responses, conditions):
            if "true" in response.choices[0].message.content.strip().lower():
                true_conditions.append(condition)

        return true_conditions

    def get_tools(self):
        return [
            {
                "name": "transfer_call",
                "description": "Transfers when the agent agrees to transfer the call",
                "parameter_definitions": {
                    "transfer_reason": {
                        "description": "The reason for transferring the call, limited to 120 characters",
                        "type": "str",
                        "required": True,
                    }
                },
            },
            {
                "name": "hangup_call",
                "description": "Hangup the call if the assistant does not think the conversation is appropriate to continue",
                "parameter_definitions": {
                    "end_reason": {
                        "description": "The reason for ending the call, limited to 120 characters",
                        "type": "str",
                        "required": True,
                    }
                },
            },
            {
                "name": "search_online",
                "description": "Searches online when the agent says they will look something up",
                "parameter_definitions": {
                    "query": {
                        "description": "The search query to be sent to the online search API",
                        "type": "str",
                        "required": True,
                    }
                },
            },
            {
                "name": "send_text",
                "description": "Triggered when the agent sends a text, only if they have been provided a valid phone number and a message to send.",
                "parameter_definitions": {
                    "to_phone": {
                        "description": "The phone number to which the text message will be sent",
                        "type": "str",
                        "required": True,
                    },
                    "message": {
                        "description": "The message to be sent, limited to 120 characters",
                        "type": "str",
                        "required": True,
                    },
                },
            },
            {
                "name": "send_email",
                "description": "Triggered when the agent sends an email, only if they have been provided a valid recipient email, a subject, and a body for the email.",
                "parameter_definitions": {
                    "recipient_email": {
                        "description": "The email address of the recipient",
                        "type": "str",
                        "required": True,
                    },
                    "subject": {
                        "description": "The subject of the email",
                        "type": "str",
                        "required": True,
                    },
                    "body": {
                        "description": "The body of the email",
                        "type": "str",
                        "required": True,
                    },
                },
            },
            {
                "name": "directly_answer",
                "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
                "parameter_definitions": {
                    "no_api_reason": {
                        "description": "The reason why an API call was not made, for instance, if a piece of information was missing or if an action was not required",
                        "type": "str",
                        "required": True,
                    }
                },
            },
        ]

    async def call_function(self, function_call: FunctionCall, agent_input: AgentInput):
        action_config = self._get_action_config(function_call.name)
        if action_config is None:
            self.logger.error(
                f"Function {function_call.name} not found in agent config, skipping"
            )
            return
        action = self.action_factory.create_action(action_config)
        params = json.loads(function_call.arguments)
        user_message_tracker = None
        if "user_message" in params:
            self.logger.info(f"User message: {params['user_message']}")
            user_message = params["user_message"]
            user_message_tracker = asyncio.Event()
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=user_message)),
                agent_response_tracker=user_message_tracker,
            )
        action_input: ActionInput
        if isinstance(action, VonagePhoneCallAction):
            assert (
                agent_input.vonage_uuid is not None
            ), "Cannot use VonagePhoneCallActionFactory unless the attached conversation is a VonageCall"
            action_input = action.create_phone_call_action_input(
                agent_input.conversation_id,
                params,
                agent_input.vonage_uuid,
                user_message_tracker,
            )
        elif isinstance(action, TwilioPhoneCallAction):
            assert (
                agent_input.twilio_sid is not None
            ), "Cannot use TwilioPhoneCallActionFactory unless the attached conversation is a TwilioCall"
            action_input = action.create_phone_call_action_input(
                agent_input.conversation_id,
                params,
                agent_input.twilio_sid,
                user_message_tracker,
            )
        else:
            action_input = action.create_action_input(
                conversation_id=agent_input.conversation_id,
                params=params,
                user_message_tracker=user_message_tracker,
            )
        event = self.interruptible_event_factory.create_interruptible_event(
            action_input, is_interruptible=action.is_interruptible
        )
        assert self.transcript is not None
        # self.transcript.add_action_start_log(
        #     action_input=action_input,
        #     conversation_id=agent_input.conversation_id,
        # )
        self.actions_queue.put_nowait(event)
        return

    async def run_nonblocking_checks(
        self, conversation_id
    ) -> Optional[Dict]:  # returns None or a dict if model should be called
        tools = self.get_tools()
        if self.agent_config.actions:
            messages, messageArray = format_command_function_completion_from_transcript(
                self.tokenizer,
                self.transcript.event_logs,
                tools,
                self.agent_config.prompt_preamble,
            )
            # print role of the latest message
            self.logger.error(f"Role was: {messageArray[-1]}")
            # tool_chat = self.prepare_chat_for_tool_check(latest_agent_response)
            self.logger.info(f"tool_chat was {messages}")
            if "Function call:" in messageArray[-1]["content"]:
                self.logger.info("Skipping tool use due to tool use.")
                return None

            async def get_response(prompt_buffer) -> str:
                self.logger.info(f"Prompt buffer: {prompt_buffer}")
                response_text = ""
                async with aiohttp.ClientSession() as session:
                    base_url = getenv("AI_API_BASE")
                    data = {
                        "model": getenv("AI_MODEL_NAME_LARGE"),
                        "prompt": prompt_buffer,
                        "stream": False,
                        "stop": ["?", "SYSTEM", "<|END_OF_TURN_TOKEN|>"],
                        "max_tokens": 500,
                        "include_stop_str_in_output": True,
                    }

                    async with session.post(
                        f"{base_url}/completions", headers=HEADERS, json=data
                    ) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            if "choices" in response_data and response_data["choices"]:
                                response_text = (
                                    response_data["choices"][0]
                                    .get("text", "")
                                    .replace("SYSTEM", "")
                                )
                        else:
                            self.logger.error(
                                f"Error while getting response from tool check: {str(response)}"
                            )
                # remove end of turn
                response_text = response_text.replace("<|END_OF_TURN_TOKEN|>", "")
                return response_text

            response = await get_response(messages)
            if not response.startswith("Action: ```json"):
                self.logger.error(f"ACTION RESULT DID NOT LOOK RIGHT: {response}")
            else:
                response = response[len("Action: ```json") :].strip()
                if response.endswith("```"):
                    response = response[:-3].strip()
                try:
                    response_data = json.loads(response)
                except json.JSONDecodeError as e:
                    self.logger.error(
                        f"JSON DECODE ERROR: {e} with response: {response}"
                    )
                    return None
                except Exception as e:
                    self.logger.error(
                        f"UNEXPECTED ERROR: {e} with response: {response}"
                    )
                    return None

                if not isinstance(response_data, list) or not response_data:
                    self.logger.error(
                        f"RESPONSE FORMAT ERROR: Expected a list with data, got: {response_data}"
                    )
                    return None

                self.logger.info(f"Response was: {response_data}")
                for tool in response_data:
                    tool_name = tool.get("tool_name")
                    tool_params = tool.get("parameters")

                    if tool_name == "directly_answer":
                        self.logger.info(
                            f"No tool, model wants to directly respond: {tool_params}"
                        )
                        self.logger.info(json.dumps(tool_params))
                        if "no_api_reason" in tool_params:
                            self.no_tool_reason = tool_params["no_api_reason"]
                        return None

                    if tool_name and tool_params is not None:
                        try:
                            while not self.can_send:
                                await asyncio.sleep(0.05)
                            await self.call_function(
                                FunctionCall(
                                    name=tool_name, arguments=json.dumps(tool_params)
                                ),
                                TranscriptionAgentInput(
                                    transcription=Transcription(
                                        message="I am doing that for you now.",
                                        confidence=1.0,
                                        is_final=True,
                                        time_silent=0.0,
                                    ),
                                    conversation_id=self.conversation_id,
                                    twilio_sid=self.twilio_sid,
                                ),
                            )
                            self.can_send = False
                            self.no_tool_reason = ""
                            return {
                                "tool_name": tool_name,
                                "tool_params": json.dumps(tool_params),
                            }

                        except Exception as e:
                            self.logger.error(f"ERROR CREATING FUNCTION CALL: {e}")
                            break  # If there's an error, we stop processing further tools
                return None
            return None

    def prepare_chat(self, latest_agent_response):
        chat = format_openai_chat_messages_from_transcript(self.transcript)[1:]
        chat[-1] = {"role": "assistant", "content": latest_agent_response}
        return chat

    def prepare_chat_for_tool_check(self, latest_agent_response):
        # Extract messages and filter out non-user and non-assistant messages
        chat = format_tool_completion_from_transcript(
            self.transcript, latest_agent_response
        )

        # remove punctuation and make lowercase using punctuation and lower
        new_chat = [
            message.translate(str.maketrans("", "", string.punctuation)).lower()
            for message in chat
        ]
        return new_chat

    def prepare_messages(self, pretty_tool_descriptions, stringified_messages):
        preamble = f"""You will be provided with a conversational transcript between a caller and the receiver's assistant. During the conversation, the assistant and the caller will either be talking about random things, discussing an action the assistant might take, the assistant might be collecting information from the the assistant has the following actions it can take: {pretty_tool_descriptions}.\nYour task is to infer, for the latest inquiry, whether the assistant has completed an action. If the assistant is about to execute an action, or is preparing to, the action is not completed and you must return 'None'. If the agent has confirmed that an action has already been completed, return the name of the action. Return a single word."""
        system_message = {"role": "system", "content": preamble}
        transcript_message = {"role": "user", "content": stringified_messages}
        return system_message, transcript_message

    def get_tool_classification(self, response, tools):
        # check to make sure there is no pending action
        if self.agent_config.pending_action == "pending":
            self.logger.info(
                "Skipping tool classification as there is a pending action"
            )
            return False, None
        tool_classification = response.choices[0].message.content.lower().strip()

        self.logger.info(f"Final tool classification to trigger: {tool_classification}")
        is_classified_tool = tool_classification in [
            tool["function"]["name"].lower() for tool in tools
        ]
        return is_classified_tool, tool_classification

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
        if self.is_first_response and self.first_response:
            self.is_first_response = False
            text = self.first_response
        else:
            chat_parameters = self.get_chat_parameters()
            chat_completion = await self.aclient.chat.completions.create(
                **chat_parameters
            )
            text = chat_completion.choices[0].message.content
        return text, False

    async def generate_completion(
        self,
        human_input: str,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        is_interrupt: bool = False,
        stream_output: bool = True,
    ) -> str:
        # if there arent equal event start and event finish wait

        digits = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        # replace all written numbers with digits
        current_index = -1
        count = 0
        # get the preamble
        preamble = self.agent_config.prompt_preamble

        if self.agent_config.language != "en-US":
            # Modify the transcript for the latest user message that matches human_input
            latest_human_message = next(
                (
                    event
                    for event in reversed(self.transcript.event_logs)
                    if event.sender == Sender.HUMAN and event.text.strip()
                ),
                None,
            )
            if latest_human_message:
                translated_message = translate_message(
                    self.logger,
                    latest_human_message.text,
                    self.agent_config.language,
                    "en-US",
                )
                latest_human_message.text = translated_message

        assert self.transcript is not None

        should_continue = None
        if self.agent_config.actions:
            should_continue = await self.run_nonblocking_checks(conversation_id)
            should_continue = should_continue
            if should_continue is not None:
                self.logger.info(f"Should continue: {should_continue}")
                # action_config = self._get_action_config(function_call.name)
                name = should_continue["tool_name"]
                params = eval(should_continue["tool_params"])
                self.logger.info(f"Name: {name}, Params: {params}")
                action_config = self._get_action_config(name)
                action = self.action_factory.create_action(action_config)
                action_input = action.create_action_input(
                    conversation_id, params, user_message_tracker=None
                )
                self.transcript.add_action_start_log(
                    action_input=action_input,
                    conversation_id=conversation_id,
                )
        # log the transcript
        self.logger.debug(f"TRANSCRIPT: {self.transcript}")
        self.logger.debug(f"COMPLETION IS RESPONDING")
        if affirmative_phrase:
            chat_parameters = self.get_completion_parameters(
                affirmative_phrase=affirmative_phrase,
                did_action=should_continue,
                reason=self.no_tool_reason,
            )
        else:
            chat_parameters = self.get_completion_parameters(
                did_action=should_continue, reason=self.no_tool_reason
            )

        prompt_buffer = chat_parameters["prompt"]
        # print number of new lines in the prompt buffer
        words = prompt_buffer.split(" ")
        new_words = []
        largest_seq = 0
        current_seq = 0
        last_digit_index = -1
        for i, word in enumerate(words):
            post = ""
            if "<|END_OF_TURN_TOKEN|>" in word:
                pre = word.split("<|END_OF_TURN_TOKEN|>")[0]
                post = "<|END_OF_TURN_TOKEN|>" + word.split("<|END_OF_TURN_TOKEN|>")[1]
                word = pre
            if word.lower() in digits:
                digit_str = str(digits.index(word.lower()))
                digit_str += post
                new_words.append(digit_str)
                current_seq += 1
                if current_seq > largest_seq:
                    largest_seq = current_seq
                    last_digit_index = i
            else:
                new_words.append(word + post)
                current_seq = 0
        if last_digit_index != -1:
            sequence_start = last_digit_index - largest_seq + 1
            insert_index = sum(len(w) + 1 for w in new_words[:sequence_start])
            if largest_seq == 10:
                number_sentence = f"(with {largest_seq} digits):"
            elif largest_seq < 10:
                number_sentence = f"(with only {largest_seq} digits (<10))"
            else:
                number_sentence = f"(with {largest_seq} digits (>10))"
            if largest_seq > 3:  # only provide nu
                prompt_buffer = (
                    " ".join(new_words[:sequence_start])
                    + f" {number_sentence} "
                    + " ".join(new_words[sequence_start:])
                )
        else:
            prompt_buffer = " ".join(new_words)

        prompt_buffer = prompt_buffer.replace("  ", " ")
        if len(prompt_buffer) == 0:
            self.logger.info("Prompt buffer is empty, returning")
            return
        latest_agent_response = ""
        if not prompt_buffer or len(prompt_buffer) < len(chat_parameters["prompt"]):
            prompt_buffer = chat_parameters["prompt"]

        async def get_response(prompt_buffer) -> str:
            # self.logger.info(f"Prompt buffer: {prompt_buffer}")
            response_text = ""
            async with aiohttp.ClientSession() as session:
                base_url = getenv("AI_API_BASE")
                data = {
                    "model": getenv("AI_MODEL_NAME_LARGE"),
                    "prompt": prompt_buffer,
                    "stream": False,
                    "stop": ["?", "SYSTEM", "<|END_OF_TURN_TOKEN|>"],
                    "max_tokens": chat_parameters.get("max_tokens", 120),
                    "include_stop_str_in_output": True,
                }

                async with session.post(
                    f"{base_url}/completions", headers=HEADERS, json=data
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if "choices" in response_data and response_data["choices"]:
                            response_text = (
                                response_data["choices"][0]
                                .get("text", "")
                                .replace("SYSTEM", "")
                            )
                    else:
                        self.logger.error(
                            f"Error while getting response from command-r: {str(response)}"
                        )
            return response_text

        # Get the full response and store it
        latest_agent_response = await get_response(prompt_buffer)
        latest_agent_response = latest_agent_response.replace(
            "<|END_OF_TURN_TOKEN|>", ""
        )
        # Run the nonblocking checks in the background
        # if self.agent_config.actions:
        #     asyncio.create_task(
        #         self.run_nonblocking_checks(latest_agent_response=latest_agent_response)
        #     )
        self.can_send = False
        return latest_agent_response, True

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
        self.logger.debug(f"CHAT IS RESPONDING")

        if self.agent_config.vector_db_config:
            try:
                vector_db_search_args = {
                    "query": self.transcript.get_last_user_message()[1],
                }

                has_vector_config_namespace = getattr(
                    self.agent_config.vector_db_config, "namespace", None
                )
                if has_vector_config_namespace:
                    vector_db_search_args[
                        "namespace"
                    ] = self.agent_config.vector_db_config.namespace.lower().replace(
                        " ", "_"
                    )

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

        # Prepare headers and data for the POST request
        data = {
            "model": chat_parameters["model"],
            "messages": chat_parameters["messages"],
            "stream": True,
            "stop": ["?", "SYSTEM"],
            "include_stop_str_in_output": True,
        }

        # Perform the POST request to the OpenAI API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{getenv('AI_API_BASE')}/chat/completions",
                headers=HEADERS,
                json=data,
                timeout=None,  # Stream endpoint; no timeout
            ) as response:
                if response.status == 200:
                    all_messages = []
                    messageBuffer = ""
                    async for line in response.content:
                        if line.strip():  # Ensure line is not just whitespace
                            try:
                                # find first and last { and } to extract the JSON
                                first_brace = line.find(b"{")
                                last_brace = line.rfind(b"}")
                                chunk = json.loads(
                                    line[first_brace : last_brace + 1].strip()
                                )
                                if "choices" in chunk and chunk["choices"]:
                                    for choice in chunk["choices"]:
                                        if (
                                            "delta" in choice
                                            and "content" in choice["delta"]
                                        ):
                                            message = choice["delta"]["content"]
                                            message.replace("SYSTEM", "")
                                            # TODO Fix numbers bug $48.25 -> $ 48 err... 25
                                            # if it contains any punctuation besides a comma, yield the buffer and the message and reset it
                                            # also check if, on split by space, it is longer than 2 words
                                            if (
                                                any(p in message for p in ".!?")
                                                and len(message.split(" ")) > 2
                                            ):
                                                messageBuffer += message
                                                all_messages.append(messageBuffer)
                                                yield messageBuffer, True
                                                messageBuffer = ""
                                            else:
                                                messageBuffer += message
                                            if (
                                                "finish_reason" in choice
                                                and choice["finish_reason"] == "stop"
                                            ):
                                                if len(messageBuffer.strip()) > 0:
                                                    all_messages.append(messageBuffer)
                                                    yield messageBuffer, True
                                                    messageBuffer = ""
                                                break
                            except json.JSONDecodeError:
                                # self.logger.error(
                                #     "JSONDecodeError: Received an empty line or invalid JSON."
                                # )
                                continue
                    # send out the last message
                    if len(messageBuffer.strip()) > 0:
                        all_messages.append(messageBuffer)
                        yield messageBuffer, True
                        messageBuffer = ""

                    if len(all_messages) > 0:
                        latest_agent_response = "".join(filter(None, all_messages))

                        await self.run_nonblocking_checks(
                            latest_agent_response=latest_agent_response
                        )
                else:
                    self.logger.error(
                        f"Error while streaming from OpenAILast: {str(response)}"
                    )