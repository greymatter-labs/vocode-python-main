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


class SaveMemoActionConfig(ActionConfig, type=ActionType.SAVE_MEMO):
    phone_number: str
    memo_content: str


class SaveMemoParameters(BaseModel):
    memo_content: str = Field(
        ...,
        description="The memo content to save for the user.",
    )


class SaveMemoResponse(BaseModel):
    status: str = Field("success", description="The status of the memo saving process.")
    memo: str = Field(..., description="The memo content saved for the user.")


class SaveMemo(
    TwilioPhoneCallAction[SaveMemoActionConfig, SaveMemoParameters, SaveMemoResponse]
):
    description: str = "Saves a memo associated with the caller's phone number."
    parameters_type: Type[SaveMemoParameters] = SaveMemoParameters
    response_type: Type[SaveMemoResponse] = SaveMemoResponse

    async def save_memo(self, to_phone, memo_content):
        # Logic to save the memo to a database or storage system should be implemented here
        # For demonstration purposes, we're logging the memo content
        logger.info(f"Saving memo for {to_phone}: {memo_content}")
        try:
            # Simulated database save operation
            # db.save(to_phone, memo_content)
            return SaveMemoResponse(
                status="success",
                memo=memo_content,
            )
        except Exception as e:
            logger.error(f"Failed to save memo for {to_phone}: {e}")
            return SaveMemoResponse(
                status="failure",
                memo="",
            )

    async def run(
        self, action_input: ActionInput[SaveMemoParameters]
    ) -> ActionOutput[SaveMemoResponse]:
        """
        Executes the action of saving a memo for the user during a call.
        :param action_input: The input parameters for the action.
        """
        save_memo_response = await self.save_memo(
            self.action_config.phone_number, action_input.params.memo_content
        )
        logger.debug(f"SaveMemo response: {save_memo_response}")

        if save_memo_response.status == "success":
            return ActionOutput(
                action_type=ActionType.SAVE_MEMO,
                response=save_memo_response,
            )
        else:
            logger.error("Failed to save memo.")
            return ActionOutput(
                action_type=ActionType.SAVE_MEMO,
                response=SaveMemoResponse(status="failure", memo=""),
            )

