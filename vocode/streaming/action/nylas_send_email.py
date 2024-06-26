from typing import Optional, Type
from pydantic import BaseModel, Field
import os
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode import getenv


class NylasSendEmailActionConfig(ActionConfig, type=ActionType.NYLAS_SEND_EMAIL):
    starting_phrase: str


class NylasSendEmailParameters(BaseModel):
    recipient_email: str = Field(..., description="The email address of the recipient.")
    body: str = Field(..., description="The body of the email.")
    subject: Optional[str] = Field(None, description="The subject of the email.")


class NylasSendEmailResponse(BaseModel):
    success: bool


class SendEmail(
    BaseAction[
        NylasSendEmailActionConfig, NylasSendEmailParameters, NylasSendEmailResponse
    ]
):
    description: str = "Sends an email using Nylas API."
    parameters_type: Type[NylasSendEmailParameters] = NylasSendEmailParameters
    response_type: Type[NylasSendEmailResponse] = NylasSendEmailResponse

    async def send_email(
        self, action_input: ActionInput[NylasSendEmailParameters]
    ) -> ActionOutput[NylasSendEmailResponse]:
        from nylas import Client

        # Initialize the Nylas client
        nylas = Client(api_key=getenv("NYLAS_API_KEY"))

        # Create the email draft
        draft = nylas.drafts.create()
        draft.body = action_input.params.body

        email_subject = action_input.params.subject
        draft.subject = email_subject if email_subject else "Email from Vocode"
        draft.to = [{"email": action_input.params.recipient_email.strip()}]

        # Send the email
        draft.send()

        return ActionOutput(
            action_type=self.action_config.type,
            response=NylasSendEmailResponse(success=True),
        )
