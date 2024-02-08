import os
import secrets
import asyncio
from typing import Type
from pydantic import BaseModel, Field
from vocode.streaming.action.phone_call_action import TwilioPhoneCallAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.models.telephony import TwilioConfig

class TransferCallActionConfig(ActionConfig, type=ActionType.TRANSFER_CALL):
    to_phone: str


class TransferCallParameters(BaseModel):
    pass


class TransferCallResponse(BaseModel):
    status: str = Field("success", description="status of the transfer")


class TransferCall(
    TwilioPhoneCallAction[
        TransferCallActionConfig, TransferCallParameters, TransferCallResponse
    ]
):
    description: str = "transfers the call. use when you need to connect the active call to another phone line."
    parameters_type: Type[TransferCallParameters] = TransferCallParameters
    response_type: Type[TransferCallResponse] = TransferCallResponse

    async def transfer_call(self, twilio_call_sid, to_phone):
        twilio_config = TwilioConfig(
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        )
        client = TwilioClient(twilio_config=twilio_config)
        conference_name = secrets.token_urlsafe(16)

        transferred_call_callback_url = f"https://{client.base_url}/handle_transferred_call_callback?telephony_call_sid={twilio_call_sid}"

        connect_to_conference_twiml_response = (
            f"<Response>"
            f"  <Dial>"
            f'    <Conference startConferenceOnEnter="true" endConferenceOnExit="true" statusCallback="{transferred_call_callback_url}" '
            f'                statusCallbackEvent="start join end" statusCallbackMethod="POST">'
            f"      {conference_name}"
            f"    </Conference>"
            f"  </Dial>"
            f"</Response>"
        )

        self.logger.info(
            f"Creating conference to agent with phone number {to_phone}"
        )

        client.twilio_client.calls.create(
            to=to_phone,
            from_=self.action_config.from_phone,  # Twilio number in our account
            twiml=connect_to_conference_twiml_response,
        )

        # Wait for a human agent to pick up before dialing the lead
        await asyncio.sleep(6)

        self.logger.info("Moving the lead into the conference")

        client.twilio_client.calls(twilio_call_sid).update(
            twiml=connect_to_conference_twiml_response
        )

        self.logger.info("Completed transferring lead to human agent.")

    async def run(
        self, action_input: ActionInput[TransferCallParameters]
    ) -> ActionOutput[TransferCallResponse]:
        twilio_call_sid = self.get_twilio_sid(action_input)

        await self.transfer_call(twilio_call_sid, self.action_config.to_phone)

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=TransferCallResponse(status="success"),
        )
