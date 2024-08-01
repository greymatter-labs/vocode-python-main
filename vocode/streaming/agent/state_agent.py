import asyncio
import json
import logging
import re
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from vocode.streaming.agent.base_agent import (
    RespondAgent,
    AgentInput,
    AgentResponseMessage,
)
from vocode.streaming.models.agent import CommandAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.state_agent_transcript import StateAgentTranscript, StateAgentTranscriptEntry
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.models.events import Sender
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from vocode import getenv
from vocode.streaming.action.phone_call_action import (
    TwilioPhoneCallAction,
    VonagePhoneCallAction,
)

from vocode.streaming.agent.utils import (
    translate_message,
)
from vocode.streaming.utils.find_sparse_subarray import find_last_sparse_subarray


class StateMachine(BaseModel):
    states: Dict[str, Any]
    initial_state_id: str


class BranchDecision(Enum):
    PAUSE = 1
    SWITCH = 2
    CONTINUE = 3


def parse_llm_dict(s):
    if isinstance(s, dict):
        return s

    s = s.replace('"', "'").replace("\n", "").replace("  ", " ").replace("','", "', '")
    result = {}
    input_string = s.strip("{}")
    pairs = input_string.split("', '")

    for pair in pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
            key = key.strip().strip("'")
            value = value.strip().strip("'")

            if value.isdigit():
                value = int(value)

            result[key] = value

    return result


def get_state(state_id_or_label: str, state_machine):
    state_id = (
        state_machine["labelToStateId"][state_id_or_label]
        if state_id_or_label in state_machine["labelToStateId"]
        and state_machine["labelToStateId"][state_id_or_label]
        else state_id_or_label
    )
    if not state_id:
        return None
    state = state_machine["states"][state_id]
    if not state:
        return None
    if state["type"] == "crossroads":
        raise Exception("crossroads state is deprecated")
    return state


def translate_to_english(
    ai_profile_language: str, text: str, logger: logging.Logger
) -> str:
    if ai_profile_language == "en-US":
        return text

    translated_message = translate_message(
        logger,
        text,
        ai_profile_language,
        "en-US",
    )
    return translated_message


async def handle_question(
    state,
    go_to_state: Callable[[str], Awaitable[Any]],
    speak_message: Callable[[Any], None],
    logger: logging.Logger,
    state_history: List[Any],
):

    await speak_message(state["question"])

    async def resume(human_input):
        logger.info(f"continuing at {state['id']} with user response {human_input}")
        return await go_to_state(get_default_next_state(state))

    return resume


async def handle_options(
    state: Any,
    go_to_state: Callable[[str], Awaitable[Any]],
    speak: Callable[[str], None],
    call_ai: Callable[[str, Dict[str, Any], Optional[str]], Awaitable[str]],
    state_machine: Any,
    get_chat_history: Callable[[], List[Tuple[str, str]]],
    logger: logging.Logger,
    state_history: List[Any],
):
    last_user_message_index = None
    last_user_message = None
    last_bot_message = state_machine["states"]["start"]["start_message"]["message"]
    action_result_after_user_spoke = None
    next_state_history = state_history + [state]

    for i, (role, msg) in enumerate(reversed(get_chat_history())):
        if last_user_message_index is None and role == "human" and msg:
            last_user_message_index = len(get_chat_history()) - i - 1
            last_user_message = msg
            break

    if last_user_message_index is not None:
        for i, (role, msg) in enumerate(
            reversed(get_chat_history()[:last_user_message_index])
        ):
            if role == "message.bot" and msg:
                last_bot_message = msg
                break

    if last_user_message_index:
        for i, (role, msg) in enumerate(
            get_chat_history()[last_user_message_index + 1 :]
        ):
            if role == "action-finish" and msg:
                action_result_after_user_spoke = msg
                break
    tool = {"condition": "[insert the number of the condition that applies]"}

    default_next_state = get_default_next_state(state)
    response_to_edge = {}
    ai_options = []
    prev_state = state_history[-1] if state_history else None
    logger.info(f"state edges {state}")
    edges = [
        edge
        for edge in state["edges"]
        if (edge.get("aiLabel") or edge.get("aiDescription"))
    ]

    logger.info(
        f"araus = {action_result_after_user_spoke} state id is {state['id']} and startingStateId is {state_machine['startingStateId']}"
    )
    if (
        state["id"] != state_machine["startingStateId"]
        and (prev_state and "question" in prev_state["id"].lower())
        and not action_result_after_user_spoke
    ):
        if len(edges) > 0:
            if edges[-1].get("aiDescription"):
                edges[-1]["aiDescription"] = (
                    f'{edges[-1]["aiDescription"]}\n\nOnly pick the following options if none of '
                    f"the above options can be applied to the current circumstance:"
                )
            else:
                edges[-1]["aiLabel"] = (
                    f'{edges[-1]["aiLabel"]}\n\nOnly pick the following options if none of the '
                    f"above options can be applied to the current circumstance:"
                )
        edges.append(
            {
                "destStateId": state_machine["startingStateId"],
                "aiLabel": "switch",
                "aiDescription": f"user no longer needs help with '{state['id'].split('::')[0]}'",
            }
        )
        if len(edges) == 1:
            edges.append(
                {
                    "destStateId": default_next_state,
                    "aiLabel": "continue",
                    "aiDescription": f"user provided an answer to the question",  # TODO: it seems to repeat the state on this
                }
            )
        else:
            edges.append(
                {
                    "destStateId": default_next_state,
                    "aiLabel": "continue",
                    "aiDescription": f"user still needs help with '{state['id'].split('::')[0]}' but no condition applies",
                }
            )
        if "question" in state["id"].split("::")[-2]:
            edges.append(
                {
                    "destStateId": default_next_state,
                    "speak": True,
                    "aiLabel": "question",
                    "aiDescription": "user seems confused or unsure",
                },
            )

    index = 0
    for index, edge in enumerate(edges):
        if "isDefault" not in edge or not edge["isDefault"]:
            response_to_edge[index] = edge
            if edge.get("aiLabel"):
                response_to_edge[edge["aiLabel"]] = edge

            ai_options.append(
                f"{index}: {edge.get('aiDescription', None) or edge.get('aiLabel', None)}"
            )

    ai_options_str = "\n".join(ai_options)
    prompt = (
        f"Bot's last statement: '{last_bot_message}'\n"
        f"User's response: '{last_user_message}'\n"
        f"{'Last tool output: ' + action_result_after_user_spoke if action_result_after_user_spoke else ''}\n\n"
        "Identify the number associated with the most fitting condition from the list below:\n"
        f"{ai_options_str}\n\n"
        "Always return a number from the above list. Return the number of the condition that best applies."
    )
    logger.info(f"AI prompt constructed: {prompt}")
    response = await call_ai(
        prompt,
        tool,
    )

    logger.info(f"Chose condition: {response}")
    try:
        response_dict = parse_llm_dict(response)
        condition = response_dict.get("condition")
        if condition is None:
            raise ValueError("No condition was provided in the response.")
        condition = int(condition)
        next_state_id = response_to_edge[condition]["destStateId"]
        if response_to_edge[condition].get("speak"):
            tool = {"response": "insert your response to the user"}
            prompt = (
                f"You last stated: '{last_bot_message}', to which the user replied: '{last_user_message}'.\n\n"
                f"You are already engaged in the following process: {state['id'].split('::')[0]}\n"
                "Right now, you are pausing the process to assist the confused user.\n"
                "Respond as follows: If the user didn't answer, politely restate the question and ask for a clear answer.\n"
                "- If the user asked a question, provide a concise answer and then pause.\n"
                "- Ensure not to suggest any actions or offer alternatives.\n"
            )
            output = await call_ai(prompt, tool)
            output = output[output.find("{") : output.find("}") + 1]
            parsed_output = parse_llm_dict(output)
            to_speak = parsed_output["response"]
            speak(to_speak)
            clarification_state = state_history[
                -1
            ].copy()  # we want to copy the last question state

            async def resume(human_input):
                logger.info(
                    f"continuing at {state['id']} with user response {human_input}"
                )
                return await go_to_state(state["id"])

            return resume, clarification_state
        return await go_to_state(next_state_id), None
    except Exception as e:
        logger.error(f"Agent chose no condition: {e}. Response was {response}")
        return await go_to_state(default_next_state), None


def get_default_next_state(state):
    if state["type"] != "options":
        return state["edge"]
    for edge in state["edges"]:
        if "isDefault" in edge and edge["isDefault"]:
            return edge["destStateId"]


class StateAgent(RespondAgent[CommandAgentConfig]):
    def __init__(
        self,
        agent_config: CommandAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        self.state_machine = self.agent_config.user_json_prompt["converted"]
        self.current_state = None
        self.resume_task = None
        self.resume = lambda _: self.handle_state(self.state_machine["startingStateId"])
        self.can_send = False
        self.conversation_id = None
        self.twilio_sid = None
        self.block_inputs = False
        self.stop = False
        self.visited_states = {self.state_machine["startingStateId"]}
        self.state_history = []
        self.chat_history = [("message.bot", self.agent_config.initial_message)]
        self.base_url = getenv("AI_API_HUGE_BASE")
        self.model = self.agent_config.model_name
        self.client = AsyncOpenAI(
            base_url=self.base_url,
        )
        self.json_transcript = StateAgentTranscript()

        self.overall_instructions = (
            self.agent_config.prompt_preamble
            + "\n"
            + self.state_machine["states"]["start"]["instructions"]
        )
        self.label_to_state_id = self.state_machine["labelToStateId"]

    def update_history(self, role, message):
        self.chat_history.append((role, message))
        self.json_transcript["entries"].append(StateAgentTranscriptEntry(role=role, message=message))
        if role == "message.bot":
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=message))
            )
        
    def get_json_transcript(self):
        return self.json_transcript

    def get_latest_bot_message(self):
        for role, message in reversed(self.chat_history):
            if isinstance(message, BaseMessage):
                if role == "message.bot" and len(message.text.strip()) > 0:
                    return message.text
            else:
                return message
        return "How can I assist you today?"

    async def generate_completion(
        self,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        human_input: str,
        is_interrupt: bool = False,
        stream_output: bool = True,
    ):
        self.update_history("human", human_input)
        self.logger.info(
            f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Lead:{human_input}"
        )
        if (
            self.resume_task
            and not self.resume_task.cancelled()
            and not self.resume_task.done()
        ):  # if something in progress, we want to move back when cancelling
            self.resume_task.cancel()
            # self.move_back_state()
            try:
                await self.resume_task
            except asyncio.CancelledError:
                self.logger.info(f"Old resume task cancelled")
        self.resume_task = asyncio.create_task(self.resume(human_input))
        await self.resume_task
        return "", True

    async def print_start_message(self, state, start: bool):
        if start and "start_message" in state:
            await self.print_message(state["start_message"])

    async def print_message(self, message):
        if message["type"] == "verbatim":
            self.update_history("message.bot", message["message"])
        else:
            guide = message["description"]
            await self.guided_response(guide)

    async def handle_state(self, state_id_or_label: str):
        start = state_id_or_label not in self.visited_states
        self.visited_states.add(state_id_or_label)
        state = get_state(state_id_or_label, self.state_machine)
        self.current_state = state
        if not state:
            return

        self.state_history.append(state)
        self.logger.info(f"Current State: {state}")

        await self.print_start_message(state, start=start)

        if state["type"] == "basic":
            return await self.handle_state(state["edge"])

        go_to_state = lambda s: self.handle_state(s)
        speak = lambda text: self.update_history("message.bot", text)
        speak_message = lambda message: self.print_message(message)
        call_ai = lambda prompt, tool, stop=None: self.call_ai(prompt, tool, stop)

        if state["type"] == "question":
            return await handle_question(
                state=state,
                go_to_state=go_to_state,
                speak_message=speak_message,
                logger=self.logger,
                state_history=self.state_history,
            )

        if state["type"] == "options":
            out, clarification_state = await handle_options(
                state=state,
                go_to_state=go_to_state,
                speak=speak,
                call_ai=call_ai,
                state_machine=self.state_machine,
                get_chat_history=lambda: self.chat_history,
                logger=self.logger,
                state_history=self.state_history,
            )
            if clarification_state:
                self.state_history.append(clarification_state)
            return out

        if state["type"] == "action":
            return await self.compose_action(state)

    async def guided_response(self, guide):
        tool = {"response": "[insert your response]"}
        message = await self.call_ai(
            f"Draft a single response to the user based on the latest chat history, taking into account the following guidance:\n'{guide}'",
            tool,
        )
        self.logger.info(f"Guided response: {message}")
        message = message[message.find("{") : message.find("}") + 1]
        self.logger.info(f"Guided response2: {message}")
        try:
            message = parse_llm_dict(message)
            self.update_history("message.bot", message["response"])
            return message.get("response")
        except Exception as e:
            self.logger.error(f"Agent could not respond: {e}")

    async def compose_action(self, state):
        action = state["action"]
        self.state_history.append(state)
        self.logger.info(f"Attempting to call: {action}")
        action_name = action["name"]
        action_config = self._get_action_config(action_name)
        if not action_config.starting_phrase or action_config.starting_phrase == "":
            action_config.starting_phrase = "One moment please..."
        to_say_start = action_config.starting_phrase
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=BaseMessage(text=to_say_start))
        )
        # self.block_inputs = True
        action_description = action["description"]
        params = action["params"]

        if params:
            dict_to_fill = {
                param_name: "[insert value]" for param_name in params.keys()
            }
            param_descriptions = "\n".join(
                [
                    f"'{param_name}': description: {params[param_name]['description']}, type: '{params[param_name]['type']})'"
                    for param_name in params.keys()
                ]
            )
            response = await self.call_ai(
                prompt=f"Based on the current conversation and the instructions provided, return a python dictionary with values inserted for these parameters: {param_descriptions}",
                tool=dict_to_fill,
            )
            self.logger.info(f"Filled params: {response}")
            response = response[response.find("{") : response.find("}") + 1]
            self.logger.info(f"Filled params: {response}")
            try:
                params = parse_llm_dict(response)
            except Exception as e:
                response = await self.call_ai(
                    prompt=f"You must return a dictionary as described. Provide the values for these parameters based on the current conversation and the instructions provided: {param_descriptions}",
                    tool=dict_to_fill,
                )
                self.logger.info(f"Filled params2: {response}")
                response = response[response.find("{") : response.find("}") + 1]
                params = parse_llm_dict(response)
        self.logger.info(f"Parsed params: {params}")

        action = self.action_factory.create_action(action_config)
        action_input: ActionInput
        if isinstance(action, TwilioPhoneCallAction):
            assert (
                self.twilio_sid is not None
            ), "Cannot use TwilioPhoneCallActionFactory unless the attached conversation is a TwilioCall"
            action_input = action.create_phone_call_action_input(
                self.conversation_id,
                params,
                self.twilio_sid,
                user_message_tracker=None,
            )
        else:
            action_input = action.create_action_input(
                conversation_id=self.conversation_id,
                params=params,
                user_message_tracker=None,
            )

        async def run_action_and_return_input(action, action_input):
            action_output = await action.run(action_input)
            pretty_function_call = (
                f"Tool Response: {action_name}, Output: {action_output}"
            )
            self.logger.info(
                f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Agent: {pretty_function_call}"
            )
            return action_input, action_output

        input, output = await run_action_and_return_input(action, action_input)
        self.block_inputs = False
        self.update_history(
            "action-finish",
            f"Action Completed: '{action_name}' completed with the following result:\ninput:'{input}'\noutput:\n{output}",
        )
        return await self.handle_state(state["edge"])

    async def call_ai(self, prompt, tool=None, stop=None):
        stop_tokens = stop if stop is not None else []
        stop_tokens.append("}")
        response_text = ""
        pretty_chat_history = "\n".join(
            [
                f"{'Bot' if role == 'message.bot' else 'Human'}: {message.text if isinstance(message, BaseMessage) else message}"
                for role, message in self.chat_history
                if (isinstance(message, BaseMessage) and len(message.text) > 0)
                or (not isinstance(message, BaseMessage) and len(str(message)) > 0)
            ]
        )
        if not tool or tool == {}:
            prompt = f"{self.overall_instructions}\n\n Given the chat history, follow the instructions.\nChat history:\n{pretty_chat_history}\n\n\nInstructions:\n{prompt}\n\nReturn a single response."
            self.logger.debug(f"prompt is: {prompt}")
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stream=True,
                temperature=0.1,
                max_tokens=500,
            )
            async for chunk in stream:
                text_chunk = chunk.choices[0].delta.content
                if text_chunk:
                    self.logger.info(f"Chunk: {chunk}")
                    response_text += text_chunk
                    if any(token in text_chunk for token in stop_tokens):
                        break
        else:
            prompt = f"Given the chat history, follow the instructions.\nChat history:\n{pretty_chat_history}\n\n\nInstructions:\n{prompt}\nYour response must always be dictionary in the following format: {str(tool)}"
            self.logger.debug(f"prompt is: {prompt}")

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.overall_instructions},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=0.1,
                max_tokens=500,
            )
            async for chunk in stream:
                text_chunk = chunk.choices[0].delta.content
                if text_chunk:
                    response_text += text_chunk
                    if any(token in text_chunk for token in stop_tokens):
                        break
        return response_text

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]

    def move_back_state(self):
        # Remove the current state
        if self.state_history:
            self.state_history.pop()

        # Find the last question state
        while self.state_history:
            previous_state = self.state_history[-1]
            if previous_state["type"] == "question":
                break
            else:
                self.state_history.pop()

        # Merge consecutive messages of the same role
        merged_messages = []
        for role, message in self.chat_history:
            current_message = (
                message.text if isinstance(message, BaseMessage) else message
            )
            if merged_messages and merged_messages[-1][0] == role:
                last_message = merged_messages[-1][1]
                if isinstance(last_message, BaseMessage):
                    merged_messages[-1] = (
                        role,
                        BaseMessage(text=f"{last_message.text} {current_message}"),
                    )
                else:
                    merged_messages[-1] = (role, f"{last_message} {current_message}")
            else:
                if isinstance(message, BaseMessage):
                    merged_messages.append((role, BaseMessage(text=current_message)))
                else:
                    merged_messages.append((role, current_message))

        # Log the merged messages for debugging
        self.logger.info(f"Merged messages: {merged_messages}")
        self.chat_history = merged_messages
        # self.chat_history = list(reversed(merged_messages))
        # Remove everything after the second to latest bot message and replace the second to latest with the removed latest
        bot_messages = [
            i
            for i, (role, _) in enumerate(merged_messages)
            if role == "bot"  # try with bot message
        ]
        if len(bot_messages) >= 2:
            second_last_user_index = bot_messages[-2]
            last_user_index = bot_messages[-1]
            merged_messages[second_last_user_index] = merged_messages[last_user_index]
            self.chat_history = merged_messages[: second_last_user_index + 1]
        else:
            self.chat_history = merged_messages
        merged_messages = []
        for role, message in self.chat_history:
            current_message = (
                message.text if isinstance(message, BaseMessage) else message
            )
            if merged_messages and merged_messages[-1][0] == role:
                last_message = merged_messages[-1][1]
                if isinstance(last_message, BaseMessage):
                    merged_messages[-1] = (
                        role,
                        BaseMessage(text=f"{last_message.text} {current_message}"),
                    )
                else:
                    merged_messages[-1] = (role, f"{last_message} {current_message}")
            else:
                if isinstance(message, BaseMessage):
                    merged_messages.append((role, BaseMessage(text=current_message)))
                else:
                    merged_messages.append((role, current_message))

        self.chat_history = merged_messages
        # if the last message is a user message and there are consecutive user messages before it that are contained in the last user message, remove the consecutive before ones
        if self.chat_history[-1][0] == "human":
            # go backwards
            for i in range(len(self.chat_history) - 2, -1, -1):
                if (
                    self.chat_history[i][0] == "human"
                    and self.chat_history[i][1].text in self.chat_history[-1][1].text
                ):
                    self.chat_history = self.chat_history[: i + 1]
                    break
                else:
                    break

        # Set the resume state
        if self.state_history:
            self.resume = lambda _: self.handle_state(self.state_history[-1]["id"])
        else:
            self.resume = lambda _: self.handle_state(
                self.state_machine["startingStateId"]
            )

    # this might not be needed.
    def restore_resume_state(self):
        if self.state_history:
            current_state = self.state_history[-1]
            if "edge" in current_state:
                self.resume = lambda _: self.handle_state(current_state["edge"])
            elif "edges" in current_state:
                for state in current_state["edges"]:
                    if "isDefault" in state and state["isDefault"]:
                        self.resume = lambda _: self.handle_state(state["destStateId"])
                        return
                self.resume = lambda _: self.handle_state(current_state[id])
            else:
                self.resume = lambda _: self.handle_state("start")
