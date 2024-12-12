import asyncio
import audioop
from dataclasses import dataclass
import json
import logging
import time
from collections import deque
from time import time_ns
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import torch
import websockets
from vocode import getenv
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
)
from vocode.streaming.utils.worker import AsyncWorker
from websockets.client import WebSocketClientProtocol
from numpy.typing import DTypeLike

PUNCTUATION_TERMINATORS = [".", "!", "?"]
MAX_SILENCE_DURATION = 2.0
NUM_RESTARTS = 5

# Silero VAD setup


model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


@dataclass
class AudioEncodingModel:
    name: str
    chunk: int
    dtype: DTypeLike
    speech_default_sampling_rate: int

    class Config:
        arbitrary_types_allowed = True


class VADWorker(AsyncWorker):

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        transcriber: "DeepgramTranscriber",
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(input_queue, output_queue)
        self.vad_buffer = b""
        # # Constants for Silero VAD
        self.WINDOWS = 3
        self.CHUNK = self.transcriber.encoding.chunk
        self.SAMPLE_RATE = self.transcriber.encoding.speech_default_sampling_rate
        self.WINDOW_SIZE = self.WINDOWS * self.SAMPLE_RATE // self.CHUNK
        self.voiced_confidences = deque([0.0] * self.WINDOWS, maxlen=self.WINDOWS)

        self.PREFIX_PADDING_MS = 150  # Minimum speech duration to trigger detection
        self.GRACE_PERIOD_MS = 300  # Grace period before resetting speech detection
        self.last_vad_output_time = 0
        self.speech_start_time = 0
        self.last_speech_time = 0
        self.logger = logger or logging.getLogger(__name__)
        self.transcriber = transcriber

    def int2float(self, sound: np.ndarray) -> np.ndarray:
        abs_max: np.ndarray = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= (
                1 / 32768 if self.transcriber.encoding.dtype == np.int16 else 1 / 128
            )
        return sound.squeeze()

    async def _run_loop(self):
        cursor_index = 0
        while True:
            try:
                cursor_index += 1
                item = await self.input_queue.get()
                chunk, timestamp, is_silence, ignore = (
                    item["chunk"],
                    item["timestamp"],
                    item["is_silence"],
                    item["ignore"],
                )
                if ignore:
                    continue
                delta = time_ns() - timestamp
                new_confidence = 0.0
                if not is_silence:
                    self.vad_buffer += chunk

                    if cursor_index % 250 == 0:
                        self.logger.debug(
                            f"VAD buffer stats: received={cursor_index}, chunk_size={len(chunk)}, buffer_size={len(self.vad_buffer)}, delta={delta/1e9:.3f}s"
                        )

                    if len(self.vad_buffer) >= self.CHUNK:
                        audio_chunk = self.vad_buffer[-self.CHUNK :]
                        self.vad_buffer = b""

                        audio_float32 = self.int2float(
                            np.frombuffer(audio_chunk, self.transcriber.encoding.dtype)
                        )
                        new_confidence = model(
                            torch.from_numpy(audio_float32), self.SAMPLE_RATE
                        ).item()
                else:
                    # Ensure the deque only keeps the last 3 confidences
                    while len(self.voiced_confidences) > 3:
                        self.logger.debug(
                            f"VAD: Removing oldest confidence: {self.voiced_confidences.popleft()}"
                        )
                        self.voiced_confidences.popleft()

                self.voiced_confidences.append(new_confidence)
                rolling_avg = sum(self.voiced_confidences) / len(
                    self.voiced_confidences
                )

                current_time = time.time()

                if rolling_avg > self.transcriber.VAD_THRESHOLD:
                    # if self.speech_start_time != 0:
                    # self.logger.debug(
                    #     f"VAD: current speech time: {current_time - self.speech_start_time}"
                    # )
                    # self.logger.debug(
                    #     f"VAD: Rolling avg: {rolling_avg}, threshold: {self.transcriber.VAD_THRESHOLD}"
                    # ) usefull for debugging

                    if self.speech_start_time == 0:
                        self.speech_start_time = current_time
                    elif (
                        current_time - self.speech_start_time
                        >= self.PREFIX_PADDING_MS / 1000
                    ):
                        self.voiced_confidences = deque(
                            [0.0] * self.WINDOWS, maxlen=self.WINDOWS
                        )
                        self.output_queue.put_nowait(
                            Transcription(
                                message="vad",
                                confidence=rolling_avg,
                                is_final=False,
                                time_silent=time_ns(),
                            )
                        )
                        self.last_vad_output_time = current_time
                    self.last_speech_time = current_time
                else:
                    # Only reset speech_start_time if we've exceeded the grace period
                    if self.last_speech_time > 0 and (
                        current_time - self.last_speech_time
                    ) > (self.GRACE_PERIOD_MS / 1000):
                        self.speech_start_time = 0
                        self.last_speech_time = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"VAD error: {str(e)}", exc_info=True)

    def send_audio(self, chunk):
        # self.logger.debug(f"vad send_audio {self.transcriber.is_muted=}")
        if not self.transcriber.is_muted:
            self.consume_nonblocking(chunk)
        else:
            self.consume_nonblocking(
                {
                    "chunk": b"",
                    "ignore": True,
                    "timestamp": time_ns(),
                    "is_silence": True,
                }
            )


class DeepgramTranscriber(BaseAsyncTranscriber[DeepgramTranscriberConfig]):
    ENCODING_MAPPING = {
        AudioEncoding.LINEAR16: AudioEncodingModel(
            name="linear16",
            dtype=np.int16,
            chunk=512,
            speech_default_sampling_rate=16_000,  # recomended by google speech to text
        ),
        AudioEncoding.MULAW: AudioEncodingModel(
            name="mulaw",
            dtype=np.int8,
            chunk=256,
            speech_default_sampling_rate=8_000,
        ),
    }

    def __init__(
        self,
        transcriber_config: DeepgramTranscriberConfig,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        vad_threshold: float = 0.35,
        volume_threshold: int = 700,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise Exception("DEEPGRAM_API_KEY not set")
        self._ended = False
        self.logger = logger or logging.getLogger(__name__)

        self.audio_cursor = 0.0
        self.VAD_THRESHOLD = vad_threshold
        self.VOLUME_THRESHOLD = volume_threshold
        self.vad_input_queue = asyncio.Queue[str]()
        self.vad_output_queue = asyncio.Queue[Transcription]()
        self.vad_worker = VADWorker(
            self.vad_input_queue, self.vad_output_queue, self, logger
        )
        self.vad_worker_task = None
        self.is_muted = False
        self.encoding = self.ENCODING_MAPPING[self.transcriber_config.audio_encoding]

    async def _run_loop(self):
        self.vad_worker_task = self.vad_worker.start()
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            self.logger.debug(f"Deepgram connection restarting, attempt {restarts}")

    def is_volume_low(self, chunk: bytes) -> bool:
        try:
            # Additional check for low volume
            if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
                volume = np.abs(np.frombuffer(chunk, self.encoding.dtype)).mean()
            else:
                # TODO: FIX ENCODING
                decoded = audioop.ulaw2lin(chunk, 2)
                volume = np.abs(np.frombuffer(decoded, self.encoding.dtype)).mean()
        except Exception as e:
            self.logger.warning(f"Error calculating volume: {e}")
            return True
        return volume < self.VOLUME_THRESHOLD

    def send_audio(self, chunk):
        if (
            self.transcriber_config.downsampling
            and self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
        ):
            self.logger.debug(
                f"downsampling {self.transcriber_config.downsampling=} {self.transcriber_config.audio_encoding =}"
            )
            chunk, _ = audioop.ratecv(
                chunk,
                2,
                1,
                self.transcriber_config.sampling_rate
                * self.transcriber_config.downsampling,
                self.transcriber_config.sampling_rate,
                None,
            )
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            chunk = np.frombuffer(chunk, self.encoding.dtype).tobytes()

        is_silence = self.is_volume_low(chunk)
        if is_silence:
            self.logger.debug(f"is_silence {is_silence=}")
        self.vad_worker.send_audio(
            {
                "chunk": chunk,
                "timestamp": time_ns(),
                "is_silence": is_silence,
                "ignore": False,
            }
        )

        super().send_audio(chunk)

    def terminate(self):
        self.input_queue.put_nowait(json.dumps({"type": "CloseStream"}))
        self._ended = True
        if self.vad_worker_task:
            self.vad_worker_task.cancel()
        super().terminate()

    def get_deepgram_url(self):
        encoding = (
            "linear16"
            if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
            else "mulaw"
        )
        assert self.transcriber_config.sampling_rate == 48000 and encoding == "linear16"
        url_params = {
            "encoding": encoding,
            "self.SAMPLE_RATE": self.transcriber_config.sampling_rate,
            "channels": 1,
            "vad_events": "false",
            "interim_results": "false",
            "filler_words": "true",
        }
        extra_params = {
            k: v
            for k, v in {
                "language": self.transcriber_config.language,
                "model": self.transcriber_config.model,
                "tier": self.transcriber_config.tier,
                "version": self.transcriber_config.version,
                "keywords": self.transcriber_config.keywords,
            }.items()
            if v is not None
        }
        if isinstance(
            self.transcriber_config.endpointing_config, PunctuationEndpointingConfig
        ):
            extra_params["punctuate"] = "true"
        url_params.update(extra_params)
        return f"wss://api.deepgram.com/v1/listen?{urlencode(url_params, doseq=True)}"

    def is_speech_final(
        self, current_buffer: str, deepgram_response: dict, time_silent: float
    ):
        transcript = deepgram_response["channel"]["alternatives"][0]["transcript"]
        if not self.transcriber_config.endpointing_config:
            return transcript and deepgram_response["speech_final"]
        elif isinstance(
            self.transcriber_config.endpointing_config, TimeEndpointingConfig
        ):
            return (
                not transcript
                and current_buffer
                and (time_silent + deepgram_response["duration"])
                > self.transcriber_config.endpointing_config.time_cutoff_seconds
            )
        elif isinstance(
            self.transcriber_config.endpointing_config, PunctuationEndpointingConfig
        ):
            return (
                transcript
                and deepgram_response["speech_final"]
                and transcript.strip()[-1] in PUNCTUATION_TERMINATORS
            ) or (
                not transcript
                and current_buffer
                and (time_silent + deepgram_response["duration"])
                > self.transcriber_config.endpointing_config.time_cutoff_seconds
            )
        raise Exception("Unsupported endpointing config")

    def calculate_time_silent(self, data: dict):
        end = data["start"] + data["duration"]
        words = data["channel"]["alternatives"][0]["words"]
        return end - words[-1]["end"] if words else data["duration"]

    async def process(self):
        self.audio_cursor = 0.0
        async with websockets.connect(
            self.get_deepgram_url(),
            extra_headers={"Authorization": f"Token {self.api_key}"},
        ) as ws:
            await asyncio.gather(self.sender(ws), self.receiver(ws))
        self.logger.debug("Terminating Deepgram transcriber process")

    async def sender(self, ws: WebSocketClientProtocol):
        while not self._ended:
            try:
                data = await asyncio.wait_for(self.input_queue.get(), 20)
                buff = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(buff).mean()
                gt_threshold = np.sum(buff > self.VOLUME_THRESHOLD)
                self.logger.debug(f"sender {volume=} {gt_threshold/len(buff)=}")
                assert data is not None and len(data) > 0

                self.audio_cursor += len(data) / (
                    self.transcriber_config.sampling_rate * 2
                )
                # self.logger.debug("sender after ")
                await ws.send(data)
            except asyncio.exceptions.TimeoutError:
                self.logger.error("Deepgram transcriber sender timeout")
                break
        self.logger.error("Terminating Deepgram transcriber sender")

    async def receiver(self, ws: WebSocketClientProtocol):
        while not self._ended:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                self.logger.error(f"receiver {data=}")

                if "is_final" not in data:
                    break
                top_choice = data["channel"]["alternatives"][0]
                self.logger.error(
                    transc := Transcription(
                        message=json.dumps(top_choice),
                        confidence=top_choice["confidence"],
                        is_final=data["is_final"],
                        time_silent=self.calculate_time_silent(data),
                    )
                )

                self.output_queue.put_nowait(transc)
                try:
                    vad_result = self.vad_output_queue.get_nowait()
                    # self.logger.debug(f"vad_result {vad_result[:10]}")
                    if vad_result:
                        self.output_queue.put_nowait(vad_result)
                except asyncio.QueueEmpty:
                    pass

            except Exception as e:
                self.logger.debug(f"Error in Deepgram receiver: {e}")
                break
        self.logger.debug("Terminating Deepgram transcriber receiver")
