import asyncio
import audioop
import json
import logging
from typing import Optional, List, Deque
from urllib.parse import urlencode
import queue
import threading
from collections import deque
import time

import websockets
from openai import AsyncOpenAI, OpenAI
from vocode import getenv
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.transcriber import (
    ClassifierEndpointingConfig,
    DeepgramTranscriberConfig,
    EndpointingConfig,
    EndpointingType,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from websockets.client import WebSocketClientProtocol

import torch
import numpy as np

PUNCTUATION_TERMINATORS = [".", "!", "?"]
MAX_SILENCE_DURATION = 2.0
NUM_RESTARTS = 5

# Silero VAD setup
torch.set_num_threads(5)
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Constants for Silero VAD
USE_INT16 = False
CHANNELS = 1
SAMPLE_RATE = 16000 if USE_INT16 else 8000
DTYPE = np.int16 if USE_INT16 else np.int8
CHUNK = 512 if USE_INT16 else 256
WINDOWS = 3
WINDOW_SIZE = WINDOWS * SAMPLE_RATE // CHUNK  # 5 seconds window
VAD_THRESHOLD = 0.75  # Adjust this threshold as needed

avg_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.deepgram.duration",
    unit="seconds",
)


class DeepgramTranscriber(BaseAsyncTranscriber[DeepgramTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: DeepgramTranscriberConfig,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config)
        self.api_key = api_key or getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set DEEPGRAM_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.is_ready = False
        self.logger = logger or logging.getLogger(__name__)
        self.audio_cursor = 0.0
        self.openai_client = OpenAI(api_key="EMPTY", base_url=getenv("AI_API_BASE"))
        self.vad_queue = queue.Queue()
        self.voiced_confidences: Deque[float] = deque([0.0] * WINDOWS, maxlen=WINDOWS)
        self.vad_thread = threading.Thread(target=self.vad_worker)
        self.vad_thread.start()
        self.vad_buffer = b""
        self.last_vad_output_time = 0

    def int2float(self, sound):
        if isinstance(sound, bytes):
            sound = np.frombuffer(sound, dtype=DTYPE)
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / 32768 if USE_INT16 else 1 / 128
        sound = sound.squeeze()
        return sound

    def vad_worker(self):
        while not self._ended:
            try:
                chunk = self.vad_queue.get(timeout=0.1)
                self.vad_buffer += chunk
                while len(self.vad_buffer) >= CHUNK:
                    audio_chunk = self.vad_buffer[:CHUNK]
                    self.vad_buffer = self.vad_buffer[CHUNK:]
                    audio_float32 = self.int2float(audio_chunk)
                    new_confidence = model(
                        torch.from_numpy(audio_float32), SAMPLE_RATE
                    ).item()
                    self.voiced_confidences.append(new_confidence)

                    # Calculate rolling average
                    rolling_avg = sum(self.voiced_confidences) / len(
                        self.voiced_confidences
                    )

                    current_time = time.time()
                    if (
                        current_time - self.last_vad_output_time >= 1
                        and rolling_avg > 0.75
                    ):
                        self.logger.debug(f"SPEAKING: {rolling_avg}")
                        self.output_queue.put_nowait(
                            Transcription(
                                message="vad",
                                confidence=rolling_avg,
                                is_final=False,
                            )
                        )
                        self.last_vad_output_time = current_time
                        # reset the rolling average
                        self.voiced_confidences = deque([0.0] * WINDOWS, maxlen=WINDOWS)
                    # else:
                    # self.logger.debug(f"SILENT: {rolling_avg}")

            except queue.Empty:
                pass
            except ValueError as e:
                self.logger.warning(f"VAD error: {str(e)}")

    async def _run_loop(self):
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            self.logger.debug(
                "Deepgram connection died, restarting, num_restarts: %s", restarts
            )

    def send_audio(self, chunk):
        if (
            self.transcriber_config.downsampling
            and self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
        ):
            chunk, _ = audioop.ratecv(
                chunk,
                2,
                1,
                self.transcriber_config.sampling_rate
                * self.transcriber_config.downsampling,
                self.transcriber_config.sampling_rate,
                None,
            )
        # Convert to int16 or int8 for Silero VAD
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            chunk = np.frombuffer(chunk, dtype=np.int16).astype(DTYPE).tobytes()
        self.vad_queue.put_nowait(chunk)  # Use put_nowait to avoid blocking
        super().send_audio(chunk)

    def terminate(self):
        terminate_msg = json.dumps({"type": "CloseStream"})
        self.input_queue.put_nowait(terminate_msg)
        self._ended = True
        self.vad_thread.join()
        super().terminate()

    def get_deepgram_url(self):
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            encoding = "linear16"
        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            encoding = "mulaw"
        url_params = {
            "encoding": encoding,
            "sample_rate": self.transcriber_config.sampling_rate,
            "channels": 1,
            "vad_events": "false",  # Disable Deepgram VAD
            "interim_results": "true",
            "filler_words": "false",
        }
        extra_params = {}
        if self.transcriber_config.language:
            extra_params["language"] = self.transcriber_config.language
        if self.transcriber_config.model:
            extra_params["model"] = self.transcriber_config.model
        if self.transcriber_config.tier:
            extra_params["tier"] = self.transcriber_config.tier
        if self.transcriber_config.version:
            extra_params["version"] = self.transcriber_config.version
        if self.transcriber_config.keywords:
            extra_params["keywords"] = self.transcriber_config.keywords
        if (
            self.transcriber_config.endpointing_config
            and self.transcriber_config.endpointing_config.type
            == EndpointingType.PUNCTUATION_BASED
        ):
            extra_params["punctuate"] = "true"
        url_params.update(extra_params)
        return f"wss://api.deepgram.com/v1/listen?{urlencode(url_params, doseq=True)}"

    def is_speech_final(
        self, current_buffer: str, deepgram_response: dict, time_silent: float
    ):
        transcript = deepgram_response["channel"]["alternatives"][0]["transcript"]

        # if it is not time based, then return true if speech is final and there is a transcript
        if not self.transcriber_config.endpointing_config:
            return transcript and deepgram_response["speech_final"]
        elif isinstance(
            self.transcriber_config.endpointing_config, TimeEndpointingConfig
        ):
            # if it is time based, then return true if there is no transcript
            # and there is some speech to send
            # and the time_silent is greater than the cutoff
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

        raise Exception("Endpointing config not supported")

    def calculate_time_silent(self, data: dict):
        end = data["start"] + data["duration"]
        words = data["channel"]["alternatives"][0]["words"]
        if words:
            return end - words[-1]["end"]
        return data["duration"]

    async def process(self):
        self.audio_cursor = 0.0
        extra_headers = {"Authorization": f"Token {self.api_key}"}

        async with websockets.connect(
            self.get_deepgram_url(), extra_headers=extra_headers
        ) as ws:

            async def sender(ws: WebSocketClientProtocol):  # sends audio to websocket
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break
                    num_channels = 1
                    sample_width = 2
                    self.audio_cursor += len(data) / (
                        self.transcriber_config.sampling_rate
                        * num_channels
                        * sample_width
                    )
                    await ws.send(data)
                self.logger.debug("Terminating Deepgram transcriber sender")

            async def receiver(ws: WebSocketClientProtocol):
                buffer = ""
                buffer_avg_confidence = 0
                num_buffer_utterances = 1
                time_silent = 0
                transcript_cursor = 0.0
                last_output_time = 0
                while not self._ended:
                    try:
                        msg = await ws.recv()
                    except Exception as e:
                        self.logger.debug(f"Got error {e} in Deepgram receiver")
                        break
                    data = json.loads(msg)
                    if (
                        not "is_final" in data
                    ):  # means we've finished receiving transcriptions
                        break
                    cur_max_latency = self.audio_cursor - transcript_cursor
                    transcript_cursor = data["start"] + data["duration"]
                    cur_min_latency = self.audio_cursor - transcript_cursor

                    avg_latency_hist.record(
                        (cur_min_latency + cur_max_latency) / 2 * data["duration"]
                    )
                    duration_hist.record(data["duration"])

                    # Log max and min latencies
                    max_latency_hist.record(cur_max_latency)
                    min_latency_hist.record(max(cur_min_latency, 0))

                    is_final = data["is_final"]
                    time_silent = self.calculate_time_silent(data)
                    top_choice = data["channel"]["alternatives"][0]
                    confidence = top_choice["confidence"]
                    current_time = time.time()
                    if current_time - last_output_time >= 1:
                        self.output_queue.put_nowait(
                            Transcription(
                                message=json.dumps(
                                    top_choice
                                ),  # since we're doing interim results, we can just send the whole data dict
                                confidence=confidence,
                                is_final=is_final,
                                time_silent=time_silent,
                            )
                        )
                        last_output_time = current_time
                self.logger.debug("Terminating Deepgram transcriber receiver")

            await asyncio.gather(sender(ws), receiver(ws))
            self.logger.debug("Terminating Deepgram transcriber receiver")
