from vocode.streaming.action.base_action import BaseAction

from vocode.streaming.models.actions import ActionConfig
from vocode.streaming.action.hangup_call import HangUpCall, HangUpCallActionConfig
from vocode.streaming.action.transfer_call import TransferCall, TransferCallActionConfig
from vocode.streaming.action.search_online import SearchOnline, SearchOnlineActionConfig
from vocode.streaming.action.send_text import SendText, SendTextActionConfig
from vocode.streaming.action.nylas_send_email import (
    SendEmail,
    NylasSendEmailActionConfig,
)
from vocode.streaming.action.get_memo import GetMemo, GetMemoActionConfig
from vocode.streaming.action.save_memo import SaveMemo, SaveMemoActionConfig


class ActionFactory:
    def create_action(self, action_config: ActionConfig) -> BaseAction:
        if isinstance(action_config, NylasSendEmailActionConfig):
            return SendEmail(action_config)
        elif isinstance(action_config, TransferCallActionConfig):
            return TransferCall(action_config)
        elif isinstance(action_config, HangUpCallActionConfig):
            return HangUpCall(action_config)
        elif isinstance(action_config, SearchOnlineActionConfig):
            return SearchOnline(action_config)
        elif isinstance(action_config, SendTextActionConfig):
            return SendText(action_config)
        elif isinstance(action_config, GetMemoActionConfig):
            return GetMemo(action_config)
        elif isinstance(action_config, SaveMemoActionConfig):
            return SaveMemo(action_config)
        else:
            raise Exception("Invalid action type")
