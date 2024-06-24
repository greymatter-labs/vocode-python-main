import asyncio
import logging
import os
from typing import Type

from pydantic import BaseModel, Field
from starlette.datastructures import FormData
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.action.phone_call_action import TwilioPhoneCallAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.models.telephony import TwilioConfig

from telephony_app.integrations.twilio.twilio_webhook_client import trigger_twilio_webhook

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ForwardCallToMoovsActionConfig(
    ActionConfig, type=ActionType.FORWARD_CALL_TO_MOOVS
):
    twilio_config: TwilioConfig
    from_phone: str  # the number that is calling us
    to_phone: str  # our inbound number
    telephony_id: str
    twilio_form_data: dict = Field(..., description="Twilio call form data")
    starting_phrase: str = Field(
        ..., description="What the agent should say when starting the action"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            FormData: lambda v: None  # Exclude FormData from serialization
        }


class ForwardCallToMoovsParameters(BaseModel):
    pass


class ForwardCallToMoovsResponse(BaseModel):
    status: str = Field(None, description="The response received from the recipient")
    # ending_phrase: str = Field(None, description="What the model should say when it has executed the function call")

class ForwardCallToMoovs(
    TwilioPhoneCallAction[
        ForwardCallToMoovsActionConfig,
        ForwardCallToMoovsParameters,
        ForwardCallToMoovsResponse,
    ]
):
    description: str = (
        "Forwards the call to all drivers within corresponding company registered on Moovs"
    )
    parameters_type: Type[ForwardCallToMoovsParameters] = (
        ForwardCallToMoovsParameters
    )
    response_type: Type[ForwardCallToMoovsResponse] = (
        ForwardCallToMoovsResponse
    )

    async def forward_call_to_moovs(self):
        return await trigger_twilio_webhook(
            to=self.action_config.to_phone,
            from_=self.action_config.from_phone,
            twilio_config=self.action_config.twilio_config,
            twilio_form_data_dict=self.action_config.twilio_form_data,
            webhook_url=os.getenv("MOOVS_CALL_FORWARDING_ENDPOINT")
        )


    async def run(
            self, action_input: ActionInput[ForwardCallToMoovsParameters]
    ) -> ActionOutput[ForwardCallToMoovsResponse]:
        await asyncio.sleep(
            6.5
        )
        response = await self.forward_call_to_moovs()
        if "error" in str(response).lower():
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ForwardCallToMoovsResponse(
                    status=f"Failed to forward {self.action_config.from_phone} to registered representatives. "
                           f"Error: {response}"
                ),
            )
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=ForwardCallToMoovsResponse(
                status=f"We have forwarded the phone call. Response: {response}",
                # ending_phrase="I have forwarded the phone call. Please hold."
            ),
        )
