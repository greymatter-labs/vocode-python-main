from __future__ import annotations

import asyncio
import json
import logging
import random
import typing
from enum import Enum
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Callable,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from opentelemetry import trace
from opentelemetry.trace import Span
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.action.phone_call_action import (
    TwilioPhoneCallAction,
    VonagePhoneCallAction,
)
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    FunctionCall,
)
from vocode.streaming.models.agent import (
    AgentConfig,
    CommandAgentConfig,
    LLMAgentConfig,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.model import TypedModel
from vocode.streaming.models.state_agent_transcript import StateAgentTranscript
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.transcriber.base_transcriber import Transcription
from vocode.streaming.utils import remove_non_letters_digits
from vocode.streaming.utils.goodbye_model import GoodbyeModel
from vocode.streaming.utils.setup_tracer import end_span, start_span_in_ctx
from vocode.streaming.utils.worker import (
    InterruptibleAgentResponseEvent,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleWorker,
)

if TYPE_CHECKING:
    from vocode.streaming.utils.state_manager import ConversationStateManager

tracer = trace.get_tracer(__name__)
AGENT_TRACE_NAME = "agent"

USE_STREAMING = False


class AgentInputType(str, Enum):
    BASE = "agent_input_base"
    TRANSCRIPTION = "agent_input_transcription"
    ACTION_RESULT = "agent_input_action_result"


class AgentInput(TypedModel, type=AgentInputType.BASE.value):
    conversation_id: str
    vonage_uuid: Optional[str]
    twilio_sid: Optional[str]
    ctx: Optional[Span] = None
    agent_response_tracker: Optional[asyncio.Event] = None

    class Config:
        arbitrary_types_allowed = True


class TranscriptionAgentInput(AgentInput, type=AgentInputType.TRANSCRIPTION.value):
    transcription: Transcription
    affirmative_phrase: Optional[str] = None


class ActionResultAgentInput(AgentInput, type=AgentInputType.ACTION_RESULT.value):
    action_input: ActionInput
    action_output: ActionOutput
    is_quiet: bool = False
    affirmative_phrase: Optional[str] = None


class AgentResponseType(str, Enum):
    BASE = "agent_response_base"
    MESSAGE = "agent_response_message"
    GENERATION_COMPLETE = "agent_response_generation_completion"
    STOP = "agent_response_stop"
    FILLER_AUDIO = "agent_response_filler_audio"


class AgentResponse(TypedModel, type=AgentResponseType.BASE.value):
    pass


class AgentResponseGenerationComplete(
    TypedModel, type=AgentResponseType.GENERATION_COMPLETE.value
):
    pass


class AgentResponseMessage(AgentResponse, type=AgentResponseType.MESSAGE.value):
    message: BaseMessage
    is_interruptible: bool = True


class AgentResponseStop(AgentResponse, type=AgentResponseType.STOP.value):
    pass


class AgentResponseFillerAudio(
    AgentResponse, type=AgentResponseType.FILLER_AUDIO.value
):
    pass


AgentConfigType = TypeVar("AgentConfigType", bound=AgentConfig)


class AbstractAgent(Generic[AgentConfigType]):
    def __init__(self, agent_config: AgentConfigType):
        self.agent_config = agent_config

    def get_agent_config(self) -> AgentConfig:
        return self.agent_config

    def update_last_bot_message_on_cut_off(self, message: str):
        """Updates the last bot message in the conversation history when the human cuts off the bot's response."""
        pass

    def get_cut_off_response(self) -> str:
        assert isinstance(self.agent_config, LLMAgentConfig) or isinstance(
            self.agent_config, CommandAgentConfig
        ), "Set cutoff response is only implemented in LLMAgent and ChatGPTAgent"
        assert self.agent_config.cut_off_response is not None
        on_cut_off_messages = self.agent_config.cut_off_response.messages
        assert len(on_cut_off_messages) > 0
        return random.choice(on_cut_off_messages).text


class BaseAgent(AbstractAgent[AgentConfigType], InterruptibleWorker):
    on_json_transcript_update: Callable[[StateAgentTranscript], None] = lambda _: None

    def __init__(
        self,
        agent_config: AgentConfigType,
        action_factory: ActionFactory = ActionFactory(),
        interruptible_event_factory: InterruptibleEventFactory = InterruptibleEventFactory(),
        logger: Optional[logging.Logger] = None,
    ):
        self.input_queue: asyncio.Queue[InterruptibleEvent[AgentInput]] = (
            asyncio.Queue()
        )
        self.output_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[AgentResponse]
        ] = asyncio.Queue()
        AbstractAgent.__init__(self, agent_config=agent_config)
        InterruptibleWorker.__init__(
            self,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            interruptible_event_factory=interruptible_event_factory,
        )
        self.action_factory = action_factory
        self.actions_queue: asyncio.Queue[InterruptibleEvent[ActionInput]] = (
            asyncio.Queue()
        )
        self.logger = logger or logging.getLogger(__name__)
        self.goodbye_model = None
        if self.agent_config.end_conversation_on_goodbye:
            self.goodbye_model = GoodbyeModel()
            self.goodbye_model_initialize_task = asyncio.create_task(
                self.goodbye_model.initialize_embeddings()
            )
        self.transcript: Optional[Transcript] = None

        self.functions = self.get_functions() if self.agent_config.actions else None
        self.is_muted = False

    def get_functions(self):
        raise NotImplementedError

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    def get_json_transcript(self) -> Optional[JsonTranscript]:
        return None

    def attach_conversation_state_manager(
        self, conversation_state_manager: ConversationStateManager
    ):
        self.conversation_state_manager = conversation_state_manager

    def set_interruptible_event_factory(self, factory: InterruptibleEventFactory):
        self.interruptible_event_factory = factory

    def get_input_queue(
        self,
    ) -> asyncio.Queue[InterruptibleEvent[AgentInput]]:
        return self.input_queue

    def get_output_queue(
        self,
    ) -> asyncio.Queue[InterruptibleAgentResponseEvent[AgentResponse]]:
        return self.output_queue

    def create_goodbye_detection_task(self, message: str) -> asyncio.Task:
        assert self.goodbye_model is not None
        return asyncio.create_task(self.goodbye_model.is_goodbye(message))


class RespondAgent(BaseAgent[AgentConfigType]):
    async def handle_generate_response(
        self,
        transcription: Transcription,
        agent_input: AgentInput,
        affirmative_phrase: Optional[str] = None,
    ) -> bool:
        conversation_id = agent_input.conversation_id
        tracer_name_start = await self.get_tracer_name_start()

        ctx = agent_input.ctx
        agent_span = start_span_in_ctx(
            name=f"{tracer_name_start}.generate_total",
            parent_span=ctx,
        )

        agent_span_first = start_span_in_ctx(
            name=f"{tracer_name_start}.generate_first",
            parent_span=ctx,
        )
        if not transcription:
            self.logger.debug("No transcription, skipping response generation")
            return False

        function_call = None
        if USE_STREAMING:
            function_call = await self.respond_with_functions_streaming(
                transcription=transcription,
                affirmative_phrase=affirmative_phrase,
                conversation_id=conversation_id,
                agent_span_first=agent_span_first,
                agent_input=agent_input,
            )
        else:
            function_call = await self.respond_with_functions(
                transcription=transcription,
                affirmative_phrase=affirmative_phrase,
                conversation_id=conversation_id,
                agent_span_first=agent_span_first,
                agent_input=agent_input,
            )

        await asyncio.sleep(0)
        # TODO: implement should_stop for generate_responses

        if function_call and self.agent_config.actions is not None:
            await self.call_function(function_call, agent_input)
        end_span(agent_span)
        return False

    async def handle_respond(
        self, transcription: Transcription, conversation_id: str
    ) -> bool:
        try:
            # I don't think we use this anywhere
            tracer_name_start = await self.get_tracer_name_start()
            with tracer.start_as_current_span(f"{tracer_name_start}.respond_total"):
                response, should_stop = await self.respond(
                    transcription.message,
                    is_interrupt=transcription.is_interrupt,
                    conversation_id=conversation_id,
                )
        except Exception as e:
            self.logger.error(f"Error while generating response: {e}", exc_info=True)
            response = None
            return True
        if response:
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=response)),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off,
            )
            return should_stop
        else:
            self.logger.debug("No response generated")
        return False

    async def process(self, item: InterruptibleEvent[AgentInput]):
        if self.is_muted:
            self.logger.debug("Agent is muted, skipping processing")
            return
        assert self.transcript is not None
        try:
            agent_input = item.payload
            if isinstance(agent_input, TranscriptionAgentInput):
                transcription = typing.cast(
                    TranscriptionAgentInput, agent_input
                ).transcription
                self.transcript.add_human_message(
                    text=transcription.message,
                    conversation_id=agent_input.conversation_id,
                )

            elif isinstance(agent_input, ActionResultAgentInput):
                self.transcript.add_action_finish_log(
                    action_input=agent_input.action_input,
                    action_output=agent_input.action_output,
                    conversation_id=agent_input.conversation_id,
                )
                if agent_input.is_quiet:
                    # Do not generate a response to quiet actions
                    self.logger.debug("Action is quiet, skipping response generation")
                    return
                transcription = Transcription(
                    message=agent_input.action_output.response.json(),
                    confidence=1.0,
                    is_final=True,
                )
                if self.agent_config.pending_action:
                    self.agent_config.pending_action = None
                    # resetting pending action
                    self.logger.debug("Resetting pending action")
                if not USE_STREAMING:
                    return
            else:
                raise ValueError("Invalid AgentInput type")

            goodbye_detected_task = None
            if self.agent_config.end_conversation_on_goodbye:
                goodbye_detected_task = self.create_goodbye_detection_task(
                    transcription.message
                )
            phrase = typing.cast(
                TranscriptionAgentInput, agent_input
            ).affirmative_phrase
            self.logger.debug("Responding to transcription")
            should_stop = False
            if not USE_STREAMING and (
                ("transcription" not in locals()) or (transcription is None)
            ):
                # transcription = Transcription(
                #     message="Is the action completed?", confidence=1.0, is_final=True
                # )
                return
            if phrase:
                should_stop = await self.handle_generate_response(
                    transcription, agent_input, phrase
                )
            else:
                should_stop = await self.handle_generate_response(
                    transcription, agent_input
                )

            if should_stop:
                self.logger.debug("Agent requested to stop")
                self.produce_interruptible_agent_response_event_nonblocking(
                    AgentResponseStop()
                )
                return
            if goodbye_detected_task:
                try:
                    goodbye_detected = await asyncio.wait_for(
                        goodbye_detected_task, 0.1
                    )
                    if goodbye_detected:
                        self.logger.debug("Goodbye detected, ending conversation")
                        self.produce_interruptible_agent_response_event_nonblocking(
                            AgentResponseStop()
                        )
                        return
                except asyncio.TimeoutError:
                    self.logger.debug("Goodbye detection timed out")
        except asyncio.CancelledError:
            pass

    def _get_action_config(self, function_name: str) -> Optional[ActionConfig]:
        if self.agent_config.actions is None:
            return None
        for action_config in self.agent_config.actions:
            if action_config.type == function_name:
                return action_config
        return None

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
                agent_input.conversation_id,
                params,
                user_message_tracker,
            )
        event = self.interruptible_event_factory.create_interruptible_event(
            action_input, is_interruptible=action.is_interruptible
        )
        assert self.transcript is not None
        self.transcript.add_action_start_log(
            action_input=action_input,
            conversation_id=agent_input.conversation_id,
        )
        self.actions_queue.put_nowait(event)

    async def get_tracer_name_start(self) -> str:
        if hasattr(self, "tracer_name_start"):
            return self.tracer_name_start
        if (
            hasattr(self.agent_config, "azure_params")
            and self.agent_config.azure_params is not None
        ):
            beginning_agent_name = self.agent_config.type.rsplit("_", 1)[0]
            engine = self.agent_config.azure_params.engine
            tracer_name_start = (
                f"{AGENT_TRACE_NAME}.{beginning_agent_name}_azuregpt-{engine}"
            )
        else:
            optional_model_name = (
                f"-{self.agent_config.model_name}"
                if hasattr(self.agent_config, "model_name")
                else ""
            )
            tracer_name_start = remove_non_letters_digits(
                f"{AGENT_TRACE_NAME}.{self.agent_config.type}{optional_model_name}"
            )
        self.tracer_name_start: str = tracer_name_start
        return tracer_name_start

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        raise NotImplementedError

    def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[
        Tuple[Union[str, FunctionCall], bool], None
    ]:  # tuple of the content and whether it is interruptible
        raise NotImplementedError

    def generate_completion(
        self,
        human_input,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        is_interrupt: bool = False,
        stream_output: bool = True,
    ) -> str:
        raise NotImplementedError

    def generate_completion_streaming(
        self,
        human_input,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        is_interrupt: bool = False,
        stream_output: bool = True,
    ) -> AsyncGenerator[
        Tuple[Union[str, FunctionCall], bool], None
    ]:  # tuple of the content and whether it is interruptible
        raise NotImplementedError

    async def respond_with_functions(
        self,
        transcription: Transcription,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        agent_span_first,
        agent_input: AgentInput,
    ) -> None:
        function_call = None
        response = await self.generate_completion(
            human_input=transcription.message,
            affirmative_phrase=affirmative_phrase if affirmative_phrase else None,
            conversation_id=conversation_id,
            is_interrupt=transcription.is_interrupt,
        )

        self.produce_interruptible_agent_response_event_nonblocking(
            AgentResponseGenerationComplete(),
            is_interruptible=False,
            agent_response_tracker=agent_input.agent_response_tracker,
        )

        end_span(agent_span_first)

        # if isinstance(response, FunctionCall):
        #     function_call = response
        # if isinstance(response[0], str):
        #     self.produce_interruptible_agent_response_event_nonblocking(
        #         AgentResponseMessage(message=BaseMessage(text=response[0])),
        #         is_interruptible=False,
        #         agent_response_tracker=agent_input.agent_response_tracker,
        #     )
        # else:
        #     self.logger.debug(
        #         "No response generated: %s of type %s",
        #         response[0],
        #         type(response),
        #     )

        return function_call

    async def respond_with_functions_streaming(
        self,
        transcription: Transcription,
        affirmative_phrase: Optional[str],
        conversation_id: str,
        agent_span_first,
        agent_input: AgentInput,
    ) -> Optional[FunctionCall]:
        function_call = None
        responses = self.generate_completion_streaming(
            human_input=transcription.message,
            affirmative_phrase=affirmative_phrase if affirmative_phrase else None,
            conversation_id=conversation_id,
            is_interrupt=transcription.is_interrupt,
        )
        is_first_response = True
        async for response, is_interruptible in responses:
            self.logger.debug(f"Generated response: {response}")
            if isinstance(response, FunctionCall):
                function_call = response
                continue
            if is_first_response:
                is_first_response = False
                end_span(agent_span_first)
            self.produce_interruptible_agent_response_event_nonblocking(
                AgentResponseMessage(message=BaseMessage(text=response)),
                is_interruptible=self.agent_config.allow_agent_to_be_cut_off
                and is_interruptible,
                agent_response_tracker=agent_input.agent_response_tracker,
            )

        return function_call
