import aiohttp
import asyncio
import io
import json
import logging
from typing import Optional
import wave

from openai import AsyncOpenAI, OpenAI
import websockets
from websockets.client import WebSocketClientProtocol
import audioop
from urllib.parse import urlencode
from vocode import getenv

from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from vocode.streaming.models.transcriber import (
    ClassifierEndpointingConfig,
    DeepgramTranscriberConfig,
    EndpointingConfig,
    EndpointingType,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.models.audio_encoding import AudioEncoding


PUNCTUATION_TERMINATORS = [".", "!", "?"]
MAX_SILENCE_DURATION = 2.0
NUM_RESTARTS = 5


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
        self.is_final = False
        self.detected_language = "en"

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
        super().send_audio(chunk)

    def terminate(self):
        terminate_msg = json.dumps({"type": "CloseStream"})
        self.input_queue.put_nowait(terminate_msg)
        self._ended = True
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
            "vad_events": "true",
            "interim_results": "true",
            "filler_words": "true",
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
        return f"wss://api.deepgram.com/v1/listen?{urlencode(url_params)}"

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
            # For shorter transcripts, check if the combined silence duration exceeds a fixed threshold
            # return (
            #     time_silent + deepgram_response["duration"]
            #     > self.transcriber_config.endpointing_config.time_cutoff_seconds
            #     if time_silent and deepgram_response["duration"]
            #     else False
            # )

        raise Exception("Endpointing config not supported")

    def calculate_time_silent(self, data: dict):
        end = data["start"] + data["duration"]
        words = data["channel"]["alternatives"][0]["words"]
        if words:
            return end - words[-1]["end"]
        return data["duration"]

    # dont need
    # async def classify_audio(self, audio_buffer):
    #     # Convert the audio buffer to a WAV file in memory
    #     with io.BytesIO() as audio_io:
    #         wav_file = wave.open(audio_io, "wb")
    #         num_channels = 1
    #         sample_width = 2
    #         wav_file.setnchannels(num_channels)
    #         wav_file.setsampwidth(sample_width)
    #         wav_file.setframerate(self.transcriber_config.sampling_rate)
    #         wav_file.writeframes(audio_buffer)
    #         wav_file.close()
    #         audio_io.seek(0)

    #         # Prepare the request to Deepgram
    #         headers = {
    #             "Authorization": f"Token {self.api_key}",
    #             "Content-Type": "audio/wav",
    #         }
    #         params = {"model": "nova-2-general", "detect_language": "true"}
    #         # Create a session if it doesn't exist
    #         if not hasattr(self, "session") or self.session.closed:
    #             self.session = aiohttp.ClientSession()
    #         async with self.session.post(
    #             "https://api.deepgram.com/v1/listen",
    #             headers=headers,
    #             params=params,
    #             data=audio_io.read(),
    #         ) as response:

    #             # Parse the response from Deepgram
    #             if response.status == 200:
    #                 response_data = await response.json()
    #                 detected_language = response_data["results"]["channels"][0][
    #                     "detected_language"
    #                 ]
    #                 language_confidence = response_data["results"]["channels"][0][
    #                     "language_confidence"
    #                 ]
    #                 # Log the detected language and confidence
    #                 self.logger.info(
    #                     f"Detected language: {detected_language} with confidence: {language_confidence}"
    #                 )
    #                 if detected_language != self.detected_language:
    #                     self.detected_language = detected_language
    #                     # set the language to the detected language
    #                     self.transcriber_config.language = detected_language
    #                     # rerun loop
    #                     self._ended = True
    #                     # sleep for 25 ms
    #                     await asyncio.sleep(0.025)
    #                     # self.is_ready = False
    #                     self.logger.info("Detected language, restarting transcriber")
    #                     self._ended = False

    #                     asyncio.create_task(self._run_loop())
    #             else:
    #                 self.logger.error(
    #                     f"Failed to classify audio with status code: {response.status}"
    #                 )

    async def process(self):
        self.audio_cursor = 0.0
        extra_headers = {"Authorization": f"Token {self.api_key}"}
        self.buffer = b""

        async with websockets.connect(
            self.get_deepgram_url(), extra_headers=extra_headers
        ) as ws:

            async def sender(ws: WebSocketClientProtocol):  # sends audio to websocket
                while not self._ended:
                    try:
                        data = await asyncio.wait_for(self.input_queue.get(), 5)
                    except asyncio.exceptions.TimeoutError:
                        break
                    await ws.send(data)

                    # num_channels = 1
                    # sample_width = 2
                    # frame_duration = len(data) / (
                    #     self.transcriber_config.sampling_rate
                    #     * num_channels
                    #     * sample_width
                    # )
                    # self.audio_cursor += frame_duration
                    # #remove equivalent amount from buffer before adding
                    # self.buffer += data
                self.logger.debug("Terminating Deepgram transcriber sender")

            async def receiver(ws: WebSocketClientProtocol):
                buffer = ""
                buffer_avg_confidence = 0
                num_buffer_utterances = 1
                time_silent = 0
                transcript_cursor = 0.0
                while not self._ended:
                    try:
                        msg = await ws.recv()
                    except Exception as e:
                        self.logger.debug(f"Got error {e} in Deepgram receiver")
                        break
                    data = json.loads(msg)
                    if data["type"] == "SpeechStarted":
                        # self.logger.debug("VAD triggered")
                        self.output_queue.put_nowait(
                            Transcription(
                                message="vad",
                                confidence=1.0,
                                is_final=False,
                            )
                        )
                        continue
                    if (
                        not "is_final" in data
                    ):  # means we've finished receiving transcriptions
                        break
                    self.is_final = data["is_final"]
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
                self.logger.debug("Terminating Deepgram transcriber receiver")

            await asyncio.gather(sender(ws), receiver(ws))
