import logging
import os

from typing import Type
from pydantic import BaseModel, Field

from vocode.streaming.action.phone_call_action import TwilioPhoneCallAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)

from telephony_app.models.call_status import CallStatus
from telephony_app.models.call_type import CallType
from telephony_app.utils.call_information_handler import (
    execute_status_update_by_telephony_id,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GetMemoActionConfig(ActionConfig, type=ActionType.GET_MEMO):
    phone_number: str


class GetMemoParameters(BaseModel):
    pass


class GetMemoResponse(BaseModel):
    status: str = Field("success", description="The status of the memo retrieval.")
    memo: str = Field(..., description="The memo content retrieved from the user.")


class GetMemo(
    TwilioPhoneCallAction[GetMemoActionConfig, GetMemoParameters, GetMemoResponse]
):
    description: str = "Retrieves a memo associated with the caller's phone number."
    parameters_type: Type[GetMemoParameters] = GetMemoParameters
    response_type: Type[GetMemoResponse] = GetMemoResponse

    async def get_memo(self, to_phone):
        # Simulated retrieval of memo content for demonstration purposes
        memo_content = f"This is a test memo for the phone number {to_phone}."
        logger.info(f"Retrieved memo for {to_phone}: {memo_content}")
        return GetMemoResponse(
            status="success",
            memo=memo_content,
        )

    async def run(
        self, action_input: ActionInput[GetMemoParameters]
    ) -> ActionOutput[GetMemoResponse]:
        """
        Retrieves a memo for the user during a call.

        :param action_input: The input parameters for the action.
        """
        get_memo_response = await self.get_memo(self.action_config.phone_number)
        logger.debug(f"GetMemo response: {get_memo_response}")

        if not get_memo_response.memo:
            logger.error(f"No memo found for {self.action_config.phone_number}")
            return ActionOutput(
                action_type=ActionType.GET_MEMO,
                response=GetMemoResponse(status="failure", memo="No memo found."),
            )

        return ActionOutput(
            action_type=ActionType.GET_MEMO,
            response=get_memo_response,
        )
