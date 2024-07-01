import logging
import httpx
from typing import Type, Dict, Any
from pydantic import BaseModel, Field

from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.action.base_action import BaseAction

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CallZapActionConfig(ActionConfig, type=ActionType.CALL_ZAP):
    zap_webhook_url: str
    starting_phrase: str


class CallZapParameters(BaseModel):
    params: Dict[str, Any] = Field(
        ..., description="The parameters to be sent to the Zapier webhook"
    )


class CallZapResponse(BaseModel):
    params: Dict[str, Any] = Field(..., description="The parameters that were sent")
    response: Dict[str, Any] = Field(
        ..., description="The response from the Zapier webhook"
    )


class CallZap(BaseAction[CallZapActionConfig, CallZapParameters, CallZapResponse]):
    description: str = (
        "calls a Zapier webhook with the provided parameters and returns the response"
    )
    parameters_type: Type[CallZapParameters] = CallZapParameters
    response_type: Type[CallZapResponse] = CallZapResponse

    async def call_zap(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls a Zapier webhook using the provided URL and parameters and returns the response.

        :param url: The Zapier webhook URL
        :param params: The parameters to be sent to the Zapier webhook
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=params)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error calling Zapier webhook: {str(e)}")
                return {"error": str(e)}

    async def run(
        self, action_input: ActionInput[CallZapParameters]
    ) -> ActionOutput[CallZapResponse]:
        params = action_input.params.params
        zap_webhook_url = action_input.action_config.zap_webhook_url

        response_content = await self.call_zap(url=zap_webhook_url, params=params)

        return ActionOutput(
            action_type=action_input.action_config.type,
            response=CallZapResponse(params=params, response=response_content),
        )
