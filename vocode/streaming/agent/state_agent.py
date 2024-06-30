import json
import logging
import re
import asyncio
import time
from typing import Any, Dict, List, Optional

from vocode.streaming.agent.base_agent import (
    RespondAgent,
    AgentInput,
    AgentResponseMessage,
)
from vocode.streaming.models.agent import CommandAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.actions import ActionInput
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


class StateMachine(BaseModel):
    states: Dict[str, Any]
    initial_state_id: str


class StateAgent(RespondAgent[CommandAgentConfig]):
    def __init__(
        self,
        agent_config: CommandAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        self.state_machine = self.agent_config.user_json_prompt["converted"]
        self.current_state = None
        self.resume = None
        self.can_send = False
        self.conversation_id = None
        self.twilio_sid = None
        self.block_inputs = False  # independent of interruptions, actions cannot be interrupted when a starting phrase is present
        self.stop = False
        self.chat_history = []
        if "medusa" in self.agent_config.model_name.lower():
            self.base_url = getenv("AI_API_HUGE_BASE")
            self.model = getenv("AI_MODEL_NAME_HUGE")
        else:
            self.base_url = getenv("AI_API_BASE")
            self.model = getenv("AI_MODEL_NAME_LARGE")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="<HF_API_TOKEN>",  # replace with your token
        )

        self.logger.info(f"State machine: {self.state_machine}")
        self.overall_instructions = (
            self.agent_config.prompt_preamble
            + "\n"
            + self.state_machine["states"]["start"]["instructions"]
        )
        self.logger.info(f"Overall instructions: {self.overall_instructions}")
        self.label_to_state_id = self.state_machine["labelToStateId"]
        #     self.state_machine.initial_state_id
        # )["instructions"]

    def update_history(self, role, message):
        self.chat_history.append((role, message))
        if role == "message.bot":
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=message))
            )

    async def generate_completion(
        self,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        human_input: Optional[str] = None,
        is_interrupt: bool = False,
        stream_output: bool = True,
    ):
        if human_input:
            self.update_history("human", human_input)
            if self.agent_config.language != "en-US":
                translated_message = translate_message(
                    self.logger,
                    human_input,
                    self.agent_config.language,
                    "en-US",
                )
                self.logger.info(
                    f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Lead:{human_input}"
                )
                human_input = translated_message
            elif self.agent_config.language == "en-US":
                self.logger.info(
                    f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Lead:{human_input}"
                )
            # make sure its not in the edge either
            if (
                self.current_state
                and self.current_state["type"] != "condition"
                and "edge" in self.current_state
                and "condition" not in self.current_state["edge"]
            ):
                last_bot_message = next(
                    (
                        msg
                        for role, msg in reversed(self.chat_history)
                        if (role == "message.bot" and msg)
                    ),
                    None,
                )
                move_on = await self.maybe_respond_to_user(
                    last_bot_message, human_input
                )
                if not move_on:
                    return "", True

        if self.resume:
            if human_input:
                await self.resume(human_input)
            else:
                await self.resume(None)
            self.resume = None  # Clear resume after it's been called
        elif not self.resume:
            if not human_input:
                self.resume = await self.handle_state("start", True)
            else:
                self.resume = await self.choose_block(state=self.current_state)
        return "", True

    async def print_start_message(self, state):
        self.logger.info(f"Current State: {state}")
        if "start_message" in state:
            start_message = state["start_message"]
            if start_message["type"] == "verbatim":
                self.update_history("message.bot", start_message["message"])
            else:
                guide = start_message["description"]
                await self.guided_response(guide)

    async def handle_state(self, state_id, start=False):
        # self.update_history("debug", f"STATE IS {state_id}")
        self.resume = None
        if not state_id:
            return
        state = self.state_machine["states"][state_id]
        self.current_state = state

        if state["type"] == "crossroads":

            async def resume(human_input):
                return await self.choose_block(state=state)

            if start:
                await self.print_start_message(state)
                return resume
            else:
                return await resume("")

        await self.print_start_message(state)

        if state["type"] == "basic":
            return await self.handle_state(state["edge"])

        if state["type"] == "question":
            question = state["question"]
            if question["type"] == "verbatim":
                self.update_history("message.bot", question["message"])
            else:
                await self.guided_response(question["description"])

            async def resume(answer):
                self.logger.info(
                    f"continuing at {state_id} with user response {answer}"
                )
                next_state_id = state["edge"]
                self.current_state = None  # Reset current state
                return await self.handle_state(next_state_id)

            return resume

        if state["type"] == "condition":
            return await self.condition_check(state=state)

        if state["type"] == "action":
            return await self.compose_action(state)

    def parse_llm_dict(self, s):
        # Remove leading/trailing whitespace and braces
        s = s.strip().strip("{}")

        # Split into key-value pairs
        pairs = re.findall(r"'([^']+)'\s*:\s*(.+?)(?=,\s*'|$)", s)

        result = {}
        for key, value in pairs:
            # Remove surrounding quotes if present
            value = value.strip()
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]

            # Try to parse as JSON if it looks like a nested structure
            if value.startswith("{") or value.startswith("["):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string if parsing fails

            result[key] = value

        return result

    async def guided_response(self, guide):
        tool = {"response": "[insert your response]"}
        message = await self.call_ai(
            f"Draft your next response to the user based on the latest chat history, taking into account the following guidance:\n'{guide}'",
            tool,
        )
        self.logger.info(f"Guided response: {message}")
        message = message[message.find("{") : message.rfind("}") + 1]
        self.logger.info(f"Guided response2: {message}")
        try:
            message = self.parse_llm_dict(message)
            self.update_history("message.bot", message["response"])
        except Exception as e:
            self.logger.error(f"Agent could not respond: {e}")

    async def choose_block(self, state=None, choose_from=None):
        if state is None:
            states = self.state_machine["states"][
                self.state_machine["startingStateId"]
            ]["edges"]
        else:
            if "edges" in state:
                states = state["edges"]
            else:
                states = [state["edge"]]
        # if choose_from is set, we will let it choose from all available
        # aadd choose_from to the states list
        if choose_from:
            states += (
                choose_from
                + self.state_machine["states"][self.state_machine["startingStateId"]][
                    "edges"
                ]
            )
        states = list(set([s.split("::")[0] for s in states]))
        # remove the start state from the list if it exists
        if "start" in states:
            states.remove("start")
        if len(states) == 1:
            next_state_id = states[0]
            next_state_id = self.label_to_state_id[next_state_id]
            return await self.handle_state(next_state_id)
        # if len is 0 then go to start
        if len(states) == 0:
            return await self.handle_state("start")
        numbered_states = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(states)])
        self.logger.info(f"Numbered states: {numbered_states}")
        streamed_choice = await self.call_ai(
            f"Choose from the available states:\n{numbered_states}\nReturn just a single number corresponding to your choice."
        )
        match = re.search(r"\d+", streamed_choice)
        chosen_int = int(match.group()) if match else 1
        next_state_id = states[chosen_int - 1]
        next_state_id = self.label_to_state_id[next_state_id]
        self.logger.info(f"Chose {next_state_id}")
        return await self.handle_state(next_state_id)

    async def condition_check(self, state):
        last_bot_message = "How can I help you today?"
        last_user_message = None
        last_tool_output = None
        last_user_message_index = None

        # Iterate through the chat history to find the last user message and its index
        for i, (role, msg) in enumerate(reversed(self.chat_history)):
            if last_user_message_index is None and role == "human" and msg:
                last_user_message_index = len(self.chat_history) - i - 1
                last_user_message = msg
                break

        # If a user message was found, iterate again to find the last bot message before the user message
        if last_user_message_index is not None:
            for i, (role, msg) in enumerate(
                reversed(self.chat_history[:last_user_message_index])
            ):
                if role == "message.bot" and msg:
                    last_bot_message = msg
                    break

        # Iterate to find the last action that comes after the last user message
        for i, (role, msg) in enumerate(
            self.chat_history[last_user_message_index + 1 :]
            if last_user_message_index is not None
            else []
        ):
            if role == "action-finish" and msg:
                last_tool_output = msg
                break
        keys = state["condition"]["conditionToStateLabel"].keys()
        keys = list(keys)

        if self.current_state and not last_tool_output:
            switch_key = f"user no longer needs help with '{self.current_state['id'].split('::')[0]}'"
            continue_key = f"user still needs help with '{self.current_state['id'].split('::')[0]}' but no condition applies"
            question_key = "user seems confused or unsure"

            keys.append(switch_key)
            keys.append(continue_key)
            if "question" in self.current_state["id"].split("::")[-2]:
                keys.append(question_key)

        choices = "\n".join([f"- '{c.lower()}'" for c in keys])
        tool = {
            "condition": "[insert the name of the condition that applies]",
        }
        prompt = (  # i was gonna move a negation of the current state's relevance to choces
            f"The bot helping with: '{self.current_state['id'].split('::')[0]}'\n"
            f"Bot's last statement: '{last_bot_message}'\n"
            f"User's response: '{last_user_message}'\n"
            f"{'Last tool output: ' + last_tool_output if last_tool_output else ''}\n\n"
            "Identify the name of the most fitting condition from the list below:\n"
            f"{choices}\n\n"
            "Always return a condition from the above list. Return the exact name of the condition that best applies."
        )
        self.logger.info(f"AI prompt constructed: {prompt}")
        response = await self.call_ai(
            prompt,
            tool,
        )
        self.logger.info(f"Chose condition: {response}")
        response = response[response.find("{") : response.rfind("}") + 1]
        try:
            response = eval(response)
            condition = response["condition"]
            condition_no_punct = "".join(
                [c for c in condition if c.isalpha() or c.isspace()]
            ).lower()
            if not last_tool_output:
                question_key_no_punct = "".join(
                    [c for c in question_key if c.isalpha() or c.isspace()]
                ).lower()
                continue_key_no_punct = "".join(
                    [c for c in continue_key if c.isalpha() or c.isspace()]
                ).lower()
                switch_key_no_punct = "".join(
                    [c for c in switch_key if c.isalpha() or c.isspace()]
                ).lower()
                response_condition_no_punct = "".join(
                    [c for c in response["condition"] if c.isalpha() or c.isspace()]
                ).lower()

                if condition_no_punct == question_key_no_punct:
                    move_on = await self.maybe_respond_to_user(
                        last_bot_message, last_user_message
                    )
                    if not move_on:
                        return
                    return await self.handle_state(state["edge"])
                if condition_no_punct == continue_key_no_punct:
                    return await self.handle_state(state["edge"])
                if condition_no_punct == switch_key_no_punct:
                    return await self.choose_block(state=None)
            for condition, next_state_label in state["condition"][
                "conditionToStateLabel"
            ].items():
                if (
                    "".join(
                        [c for c in condition if c.isalpha() or c.isspace()]
                    ).lower()
                    == response_condition_no_punct
                ):
                    next_state_id = self.label_to_state_id[next_state_label]
                    return await self.handle_state(next_state_id)
            return await self.handle_state(state["edge"])
        except Exception as e:
            self.logger.error(f"Agent chose no condition: {e}")
            return await self.handle_state(state["edge"])

    async def compose_action(self, state):
        action = state["action"]
        self.logger.info(f"Attempting to call: {action}")
        action_name = action["name"]
        action_config = self._get_action_config(action_name)
        if not action_config.starting_phrase or action_config.starting_phrase == "":
            action_config.starting_phrase = "Starting action..."
        # the value of starting_phrase is the message it says during the action
        to_say_start = action_config.starting_phrase
        # if not self.streamed and not self.agent_config.use_streaming:
        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseMessage(message=BaseMessage(text=to_say_start))
        )
        # if we're using streaming, we need to block inputs until the tool calls are done
        self.block_inputs = True
        action_description = action["description"]
        params = action["params"]
        dict_to_fill = {param_name: "[insert value]" for param_name in params.keys()}
        param_descriptions = "\n".join(
            [
                f"'{param_name}': description: {params[param_name]['description']}, type: '{params[param_name]['type']})'"
                for param_name in params.keys()
            ]
        )
        response = await self.call_ai(
            prompt=f"Provide the values for these parameters based on the current conversation and the instructions provided: {param_descriptions}",
            tool=dict_to_fill,
        )
        self.logger.info(f"Filled params: {response}")
        response = response[response.find("{") : response.rfind("}") + 1]
        self.logger.info(f"Filled params: {response}")
        try:
            params = eval(response)
        except Exception as e:
            response = await self.call_ai(
                prompt=f"You must return a dictionary as described. Provide the values for these parameters based on the current conversation and the instructions provided: {param_descriptions}",
                tool=dict_to_fill,
            )
            self.logger.info(f"Filled params2: {response}")
            response = response[response.find("{") : response.rfind("}") + 1]
            params = eval(response)

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

        # check if starting_phrase exists and has content
        # TODO handle if it doesnt have one

        async def run_action_and_return_input(action, action_input):
            action_output = await action.run(action_input)
            # also log the output
            pretty_function_call = (
                f"Tool Response: {action_name}, Output: {action_output}"
            )
            self.logger.info(
                f"[{self.agent_config.call_type}:{self.agent_config.current_call_id}] Agent: {pretty_function_call}"
            )
            return action_input, action_output

        # accumulate the tasks so we dont wait on each one sequentially
        input, output = await run_action_and_return_input(action, action_input)
        self.block_inputs = False
        # TODO: HERE we need to actually run it
        self.update_history(
            "action-finish",
            f"Action Completed: '{action_name}' completed with the following result:\ninput:'{input}'\noutput:\n{output}",
        )
        # it seems to continue on an on
        if state["edge"] == "start":
            self.current_state = None
            return {}
        else:
            return await self.handle_state(state["edge"])

    async def maybe_respond_to_user(self, question, user_response):
        continue_tool = {
            "[either 'PAUSE' or 'CONTINUE' or 'SWITCH']": "[insert your response to the user]",
        }
        if self.current_state and self.current_state["id"]:
            current_workflow = self.current_state["id"].split("::")[0]
        else:
            current_workflow = "start of conversation"
        # if its a conditional, get the conditions
        if self.current_state and self.current_state["type"] == "condition":
            conditions = self.current_state["condition"]["conditionToStateLabel"].keys()
            conditions = "\n".join([f"'{c}'" for c in conditions])
        additional_info = (
            f"Decide what to do next. If the user's response aligns with any of the following, 'CONTINUE' the current process:\n{conditions}"
            if self.current_state and self.current_state["type"] == "condition"
            else ""
        )
        prompt = (
            f"You last stated: '{question}', to which the user replied: '{user_response}'.\n\n"
            f"You are already engaged in the following process: {current_workflow}\n"
            f"{additional_info}\n\n"
            "Now, to best assist the user, decide whether you should:\n"
            "- 'CONTINUE' the process to the next step\n"
            "   Response: Proceed with the next step of the current workflow without additional commentary\n"
            "- 'PAUSE' the current process if the user asked a question of their own\n"
            "   Response: If the user didn't answer, politely restate the question and ask for a clear answer.\n"
            "             If the user asked a question, provide a concise answer and then pause.\n"
            "             Ensure not to suggest any actions or offer alternatives.\n"
            "- 'SWITCH' to a different process if current process is not relevant or the user changes the subject\n"
            "   Response: Switch to a different process without additional commentary\n"
        )
        self.logger.info(f"Prompt: {prompt}")
        # we do not care what it has to say when switching or continuing
        output = await self.call_ai(prompt, continue_tool, stop=["CONTINUE", "SWITCH"])
        self.logger.info(f"Output: {output}")
        if "CONTINUE" in output:
            return True
        if "SWITCH" in output:
            # When switching, go directly to crossroads
            self.resume = await self.choose_block(state=None)

            # self.update_history("message.bot", question)
            return False
        output = output[output.find("{") : output.rfind("}") + 1]
        output = eval(output.replace("'None'", "'none'").replace("None", "'none'"))
        if "PAUSE" in output:
            self.update_history("message.bot", output["PAUSE"])

            async def resume(user_response):
                # Handle the user's response and move to the next state
                next_state_id = self.current_state["edge"]
                self.current_state = None  # Reset current state
                return await self.handle_state(next_state_id)

            self.resume = resume
            return False
        self.logger.info(f"falling back with {output}")
        return True

    async def call_ai(self, prompt, tool=None, stop=None):
        # self.client = AsyncOpenAI(
        #     base_url=self.base_url,
        #     api_key="<HF_API_TOKEN>",  # replace with your token
        # )
        if not tool:
            # self.update_history("message.instruction", prompt)
            # return input(f"[message.bot] $: ")
            chat_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.overall_instructions},
                    {
                        "role": "user",
                        "content": f"Given the chat history, follow the instructions.\nChat history:\n{self.chat_history}\n\n\nInstructions:\n{prompt}",
                    },
                ],
                stream=False,
                stop=stop if stop is not None else [],
                temperature=0.1,
                max_tokens=500,
            )
            return chat_completion.choices[0].message.content
        else:
            # log it
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Tool: {tool}")

            # self.update_history("message.instruction", prompt)
            chat_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.overall_instructions},
                    {
                        "role": "user",
                        "content": f"Given the chat history, follow the instructions.\nChat history:\n{self.chat_history}\n\n\nInstructions:\n{prompt}\nYour response must always be dictionary in the following format: {tool}",
                    },
                ],
                stream=False,
                stop=stop if stop is not None else [],
                temperature=0.1,
                max_tokens=500,
            )
            return chat_completion.choices[0].message.content

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]
