from __future__ import annotations

import asyncio
from fastapi import WebSocket
import numpy as np
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.models.websocket import AudioMessage, TranscriptMessage
from vocode.streaming.models.transcript import (
    StateAgentTranscriptEvent,
    TranscriptEvent,
)


def convert_linear16_to_pcm(linear16_audio: bytes) -> bytes:
    audio_array = (
        np.frombuffer(linear16_audio, dtype=np.int16).astype(np.float32) / 32768.0
    )
    return audio_array.tobytes()


class WebsocketOutputDevice(BaseOutputDevice):
    def __init__(
        self,
        ws: WebSocket,
        sampling_rate: int,
        audio_encoding: AudioEncoding,
        into_pcm: bool = False,
    ):
        super().__init__(sampling_rate, audio_encoding)
        self.ws = ws
        self.active = False
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.into_pcm = into_pcm

    def start(self):
        self.active = True
        self.process_task = asyncio.create_task(self.process())

    def mark_closed(self):
        self.active = False

    async def process(self):
        while self.active:
            message = await self.queue.get()
            await self.ws.send_text(message)

    async def consume_nonblocking(self, chunk: bytes):
        if self.active:
            assert (
                self.audio_encoding == AudioEncoding.LINEAR16
            ), "Only Linear16 is supported for now"
            audio_message = AudioMessage.from_bytes(chunk)

            await self.queue.put(audio_message.json())

    def consume_transcript(self, event: TranscriptEvent | StateAgentTranscriptEvent):
        if self.active:
            if isinstance(event, StateAgentTranscriptEvent):
                transcript_message = event.transcript
            elif isinstance(event, TranscriptEvent):
                transcript_message = TranscriptMessage.from_event(event)
            self.queue.put_nowait(transcript_message.json())

    def terminate(self):
        self.process_task.cancel()

    async def clear(self):
        while not self.queue.empty():
            await self.queue.get()
