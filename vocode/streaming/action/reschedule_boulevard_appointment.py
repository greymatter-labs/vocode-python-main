import logging
import os
from datetime import datetime
from typing import Optional, Type
from pydantic import BaseModel, Field
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)
from vocode.streaming.action.base_action import BaseAction
from telephony_app.integrations.boulevard.boulevard_client import (
    CURRENT_BOULEVARD_CREDENTIALS,
    get_available_reschedule_times,
    get_time_slot,
    reschedule_appointment,
    retrieve_next_appointment_by_phone_number,
    find_nearest_available_times,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RescheduleBoulevardAppointmentActionConfig(
    ActionConfig, type=ActionType.RESCHEDULE_BOULEVARD_APPOINTMENT
):
    appointment_to_reschedule: dict
    business_id: str
    
class RescheduleBoulevardAppointmentParameters(BaseModel):
    available_times: list[dict]
    target_time: str

class RescheduleBoulevardAppointmentResponse(BaseModel):
    succeeded: bool
    new_appointment_time: Optional[str] = None

class RescheduleBoulevardAppointment(
    BaseAction[
        RescheduleBoulevardAppointmentActionConfig,
        RescheduleBoulevardAppointmentParameters,
        RescheduleBoulevardAppointmentResponse,
    ]
):
    description: str = """Reschedules an appointment on Boulevard.
    Note: this action only currently supports rescheduling for the next appointment."""
    parameters_type: Type[RescheduleBoulevardAppointmentParameters] = RescheduleBoulevardAppointmentParameters
    response_type: Type[RescheduleBoulevardAppointmentResponse] = RescheduleBoulevardAppointmentResponse

    async def run(
        self, action_input: ActionInput[RescheduleBoulevardAppointmentParameters]
    ) -> ActionOutput[RescheduleBoulevardAppointmentResponse]:        
        if not self.action_config.appointment_to_reschedule:
            logger.error("No upcoming appointment found.")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=RescheduleBoulevardAppointmentResponse(succeeded=False)
            )

        selected_time = datetime.strptime(action_input.params.target_time, "%H:%M").strftime("%-I:%M %p")
        selected_slot = get_time_slot(action_input.params.available_times, selected_time)
        if not selected_slot.get('id', ''):
            logger.error(f"The selected time slot {selected_time} is not available for rescheduling.")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=RescheduleBoulevardAppointmentResponse(succeeded=False)
            )
            
        response = reschedule_appointment(
            appointment_id=self.action_config.appointment_to_reschedule['id'],
            bookable_time_id=selected_slot['id'],
            business_id=CURRENT_BOULEVARD_CREDENTIALS.business_id,
            env=os.getenv('BOULEVARD_ENV', 'dev'),
            send_notification=True
        )

        if response and response.get("data") and response["data"].get("appointmentReschedule"):
            new_appointment = response["data"]["appointmentReschedule"]["appointment"]
            logger.info(f"Appointment rescheduled successfully. New appointment ID: {new_appointment['id']}")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=RescheduleBoulevardAppointmentResponse(
                    succeeded=True,
                    new_appointment_time=action_input.params.target_time
                )
            )
        else:
            logger.error("Failed to reschedule the appointment.")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=RescheduleBoulevardAppointmentResponse(succeeded=False)
            )

