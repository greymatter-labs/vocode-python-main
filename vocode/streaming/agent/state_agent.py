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
        # self.client = AsyncOpenAI(
        #     base_url=self.base_url,
        #     api_key="<HF_API_TOKEN>",  # replace with your token
        # )
        self.logger.info(f"Hellpo")  # I left off with it not parsing the state machien

        self.logger.info(f"State machine: {self.state_machine}")
        self.overall_instructions = self.state_machine["states"]["start"][
            "instructions"
        ]
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
            last_bot_message = next(
                (
                    msg[1]
                    for msg in reversed(self.chat_history)
                    if msg[0] == "message.bot" and msg[1] and len(msg[1].strip()) > 0
                ),
                "",
            )
            move_on = await self.maybe_respond_to_user(last_bot_message, human_input)
            if not move_on:
                self.update_history("message.bot", last_bot_message)
                return (
                    "",
                    True,
                )  # TODO: it repeats itself here, if maybe ends up saying something, no need to say it here with update
        if self.resume and human_input:
            await self.resume(human_input)
        elif not self.resume and not human_input:
            self.resume = await self.handle_state("start", True)
        else:
            self.resume = await self.choose_block(state=self.current_state)
        return "", True

    async def print_start_message(self, state):
        start_message = state["start_message"]
        if start_message:
            if start_message["type"] == "verbatim":
                self.update_history("message.bot", start_message["message"])
            else:
                guide = start_message["description"]
                await self.guided_response(guide)

    async def handle_state(self, state_id, start=False):
        self.update_history("debug", f"STATE IS {state_id}")
        if not state_id:
            return
        state = self.state_machine["states"][state_id]
        self.current_state = state

        if state["type"] == "crossroads":

            async def resume(human_input):
                self.resume = None
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
            self.update_history("message.bot", state["question"]["prompt"])

            async def resume(answer):
                self.update_history(
                    "debug", f"continuing at {state_id} with user response {answer}"
                )
                return await self.handle_state(state["edge"])

            return resume

        if state["type"] == "condition":
            return await self.condition_check(state=state)

        if state["type"] == "action":
            return await self.compose_action(state)

    async def guided_response(self, guide):
        self.update_history("message.instruction", f"Your prompt is {guide}")
        message = await self.call_ai("What do you want to say?")
        self.update_history("message.bot", message)

    async def choose_block(self, state=None):
        if state is None:
            states = self.state_machine["states"][
                self.state_machine["startingStateId"]
            ]["edges"]
        else:
            if "edges" in state:
                states = state["edges"]
            else:
                states = [state["edge"]]
        if len(states) == 1:
            next_state_id = states[0]
            return await self.handle_state(next_state_id)
        numbered_states = "\n".join([f"{i + 1}. {s}" for i, s in enumerate(states)])
        streamed_choice = await self.call_ai(
            f"Choose from the available states:\n{numbered_states}\nReturn just a single number corresponding to your choice."
        )
        match = re.search(r"\d+", streamed_choice)
        chosen_int = int(match.group()) if match else 1
        next_state_id = states[chosen_int - 1]
        self.update_history("debug", f"The bot chose {chosen_int}, aka {next_state_id}")
        return await self.handle_state(next_state_id)

    async def condition_check(self, state):
        self.update_history("debug", "Running condition check")
        choices = "\n".join(
            [f"'{c}'" for c in state["condition"]["conditionToStateId"].keys()]
        )
        self.update_history("debug", f"Choices: {choices}")
        tool = {"choice": "[insert the user's choice]"}
        response = await self.call_ai(
            f"{state['condition']['prompt']}. Choices: {list(state['condition']['conditionToStateId'].keys())}\nChoose the right choice.",
            tool,
        )
        response = response[response.find("{") : response.rfind("}") + 1]
        response = eval(response)
        for condition, next_state_id in state["condition"][
            "conditionToStateId"
        ].items():
            if condition == response["choice"]:
                return await self.handle_state(next_state_id)
            else:
                self.update_history(
                    "debug", f"condition {condition} != response {response}"
                )

    async def compose_action(self, state):
        action = state["action"]
        self.logger.info(f"Attempting to call: {action}")
        action_name = action["name"]
        action_description = action["description"]
        params = action["params"]
        dict_to_fill = {
            param_name: f"{params[param_name]['description']} of type: {params[param_name]['type']}"
            for param_name in params.keys()
        }
        response = await self.call_ai(
            f"Return the parameters for the function.", dict_to_fill
        )
        self.logger.info(f"Attempting to call with params: {response}")
        self.update_history(
            "action-start",
            f"Running action {action_name} with params response {response}",
        )
        # TODO: HERE we need to actually run it
        self.update_history("action-finish", f"Action result: <fake>")
        return await self.handle_state(state["edge"])

    async def maybe_respond_to_user(self, question, user_response):
        continue_tool = {
            "[either 'immediate' or 'workflow']": "[insert your response to the user]",
        }
        prompt = f"In the current workflow, you stated: '{question}', and the user replied: '{user_response}'.\n\nDecide if you should proceed with the workflow to assist the user, or if an immediate response followed by a repetition of your statement is required to obtain a clear answer."
        output = await self.call_ai(prompt, continue_tool, stop=["workflow"])
        self.logger.info(f"Output: {output}")
        if "workflow" in output:
            return True
        output = output[output.find("{") : output.rfind("}") + 1]
        output = eval(output.replace("'None'", "'none'").replace("None", "'none'"))
        if "immediate" in output:
            self.update_history("message.bot", output["immediate"])
            return False
        # self.update_history("message.bot", question)
        return False

    async def call_ai(self, prompt, tool=None, stop=None):
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="<HF_API_TOKEN>",  # replace with your token
        )
        if not tool:
            self.update_history("message.instruction", prompt)
            # return input(f"[message.bot] $: ")
            chat_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.overall_instructions},
                    {
                        "role": "user",
                        "content": f"Given the chat history, follow the instructions.\nChat history:\n{self.chat_history}\nInstructions:\n{prompt}",
                    },
                ],
                stream=False,
                stop=stop if stop is not None else [],
                max_tokens=500,
            )
            return chat_completion.choices[0].message.content
        else:
            self.update_history("message.instruction", prompt)
            chat_completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.overall_instructions},
                    {
                        "role": "user",
                        "content": f"Given the chat history, follow the instructions.\nChat history:\n{self.chat_history}\nInstructions:\n{prompt}\nRespond with a dictionary in the following format: {tool}",
                    },
                ],
                stream=False,
                stop=stop if stop is not None else [],
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
