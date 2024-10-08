import asyncio
import os
from typing import Dict, Type

import aiohttp
from aiohttp import BasicAuth
from pydantic import BaseModel, Field
from vocode.streaming.action.phone_call_action import TwilioPhoneCallAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.models.telephony import TwilioConfig


class TransferCallActionConfig(ActionConfig, type=ActionType.TRANSFER_CALL):
    twilio_config: TwilioConfig
    starting_phrase: str
    from_phone: str


class TransferCallParameters(BaseModel):
    phone_number_to_transfer_to: str = Field(
        ...,
        description="The phone number to transfer the call to",
    )


class TransferCallResponse(BaseModel):
    status: str = Field("success", description="status of the transfer")


class TransferCall(
    TwilioPhoneCallAction[
        TransferCallActionConfig, TransferCallParameters, TransferCallResponse
    ]
):
    description: str = (
        "transfers the call. use when you need to connect the active call to another phone line."
    )
    parameters_type: Type[TransferCallParameters] = TransferCallParameters
    response_type: Type[TransferCallResponse] = TransferCallResponse

    async def transfer_call(self, twilio_call_sid, to_phone, from_phone):
        twilio_account_sid = self.action_config.twilio_config.account_sid
        twilio_auth_token = self.action_config.twilio_config.auth_token

        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_account_sid}/Calls/{twilio_call_sid}.json"

        twiml_data = (
            f"<Response><Dial callerId='{from_phone}'>{to_phone}</Dial></Response>"
        )

        payload = {"Twiml": twiml_data}

        auth = BasicAuth(twilio_account_sid, twilio_auth_token)

        async with aiohttp.ClientSession(auth=auth) as session:
            async with session.post(url, data=payload) as response:
                if response.status != 200:
                    print(await response.text())
                    raise Exception("failed to update call")
                else:
                    return await response.json()

    async def run(
        self, action_input: ActionInput[TransferCallParameters]
    ) -> ActionOutput[TransferCallResponse]:
        twilio_call_sid = self.get_twilio_sid(action_input)

        await asyncio.sleep(
            3.5
        )  # to provide small gap between speaking and transfer ring

        if not hasattr(self.action_config, "from_phone"):
            raise ValueError("'from_phone' is not set in the action_config")

        await self.transfer_call(
            twilio_call_sid=twilio_call_sid,
            to_phone=action_input.params.phone_number_to_transfer_to,
            from_phone=self.action_config.from_phone,
        )

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=TransferCallResponse(status="success"),
        )
