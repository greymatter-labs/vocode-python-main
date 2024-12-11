from abc import ABC
from vocode.streaming.models.audio_encoding import AudioEncoding


class BaseOutputDevice(ABC):
    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

    def start(self):
        pass

    async def consume_nonblocking(self, chunk: bytes):
        raise NotImplementedError

    def maybe_send_mark_nonblocking(self, message):
        pass

    def terminate(self):
        pass

    async def clear(self):
        raise NotImplementedError

    async def process(self):
        raise NotImplementedError
