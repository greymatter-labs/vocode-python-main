import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Type
from pydantic import BaseModel
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.action.base_action import BaseAction
from telephony_app.integrations.boulevard.boulevard_client import (
    get_app_credentials,
    get_available_reschedule_times,
    parse_times,
)
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CheckBoulevardRescheduleAvailabilityActionConfig(
    ActionConfig, type=ActionType.CHECK_BOULEVARD_RESCHEDULE_AVAILABILITY
    ):
    business_id: str
    appointment_to_reschedule: dict

class CheckBoulevardRescheduleAvailabilityParameters(BaseModel):
    pass

class CheckBoulevardRescheduleAvailabilityResponse(BaseModel):
    available_times: List[str]
    message: str

class CheckBoulevardRescheduleAvailability(
    BaseAction[
        CheckBoulevardRescheduleAvailabilityActionConfig,
        CheckBoulevardRescheduleAvailabilityParameters,
        CheckBoulevardRescheduleAvailabilityResponse,
    ]
):
    description: str = """Checks for available reschedule times on Boulevard for a specific date. 
    NOTE: This action is currently only supported for the next appointment for a given phone number."""
    
    parameters_type: Type[CheckBoulevardRescheduleAvailabilityParameters] = CheckBoulevardRescheduleAvailabilityParameters
    response_type: Type[CheckBoulevardRescheduleAvailabilityResponse] = CheckBoulevardRescheduleAvailabilityResponse

    async def run(
        self, action_input: ActionInput[CheckBoulevardRescheduleAvailabilityParameters]
    ) -> ActionOutput[CheckBoulevardRescheduleAvailabilityResponse]:

        if not self.action_config.appointment_to_reschedule:
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=CheckBoulevardRescheduleAvailabilityResponse(
                    available_times=[],
                    message=f"No appointment to reschedule."
                ),
            )

        requested_date = self.action_config.appointment_to_reschedule.get('startAt', '')
        try:
            parsed_date = date_parser.parse(requested_date)
            formatted_appointment_date = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=CheckBoulevardRescheduleAvailabilityResponse(
                    available_times=[],
                    message=f"Invalid date format. Please provide the date in the format 'Month DD, YYYY' (e.g., 'September 08, 2024')."
                ),
            )
            
        available_times = get_available_reschedule_times(
            appointment_id=self.action_config.appointment_to_reschedule.get('id', ''),
            business_id=self.action_config.business_id,
            date=formatted_appointment_date,
            env=os.getenv(key="ENV", default="dev")
        )

        if not available_times:
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=CheckBoulevardRescheduleAvailabilityResponse(
                    available_times=[],
                    message=f"No available times for rescheduling on {requested_date}."
                ),
            )

        parsed_times = parse_times(available_times)
        
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=CheckBoulevardRescheduleAvailabilityResponse(
                available_times=parsed_times,
                message=f"Available times for rescheduling on {requested_date}: {', '.join(parsed_times)}."
            ),
        )
