import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import random
import re
from typing import Any, List, Optional, Tuple
from xml.etree import ElementTree
import aiohttp
from vocode import getenv
from opentelemetry.context.context import Context

from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage, SSMLMessage

from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    FILLER_PHRASES,
    AFFIRMATIVE_PHRASES,
    FILLER_AUDIO_PATH,
    AFFIRMATIVE_AUDIO_PATH,
    FillerAudio,
    encode_as_wav,
    tracer,
)
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, SynthesizerType
from vocode.streaming.models.audio_encoding import AudioEncoding

import azure.cognitiveservices.speech as speechsdk


NAMESPACES = {
    "mstts": "https://www.w3.org/2001/mstts",
    "": "https://www.w3.org/2001/10/synthesis",
}

ElementTree.register_namespace("", NAMESPACES[""])
ElementTree.register_namespace("mstts", NAMESPACES["mstts"])


class WordBoundaryEventPool:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(
            {
                "text": event.text,
                "text_offset": event.text_offset,
                "audio_offset": (event.audio_offset + 5000) / (10000 * 1000),
                "boudary_type": event.boundary_type,
            }
        )

    def get_events_sorted(self):
        return sorted(self.events, key=lambda event: event["audio_offset"])


class AzureSynthesizer(BaseSynthesizer[AzureSynthesizerConfig]):
    OFFSET_MS = 100

    def __init__(
        self,
        synthesizer_config: AzureSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        azure_speech_key: Optional[str] = None,
        azure_speech_region: Optional[str] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        azure_endpoint_id: Optional[str] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)
        # Instantiates a client

        azure_speech_key = (
            azure_speech_key
            or getenv("AZURE_SPEECH_KEY")
            or synthesizer_config.azure_speech_key
        )
        azure_speech_region = (
            azure_speech_region
            or getenv("AZURE_SPEECH_REGION")
            or synthesizer_config.azure_speech_region
        )
        azure_endpoint_id = (
            azure_endpoint_id
            or getenv("AZURE_ENDPOINT_ID")
            or synthesizer_config.azure_endpoint_id
        )
        if not azure_speech_key:
            raise ValueError(
                "Please set AZURE_SPEECH_KEY environment variable or pass it as a parameter"
            )
        if not azure_speech_region:
            raise ValueError(
                "Please set AZURE_SPEECH_REGION environment variable or pass it as a parameter"
            )
        speech_config = speechsdk.SpeechConfig(
            subscription=azure_speech_key, region=azure_speech_region
        )

        speech_config.speech_synthesis_voice_name = "PlaygroundLiteNeural"
        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            if self.synthesizer_config.sampling_rate == 44100:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == 48000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == 24000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == 16000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == 8000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm
                )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
            )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        if azure_endpoint_id and "neural" in self.synthesizer_config.voice_name.lower():
            speech_config.endpoint_id = azure_endpoint_id

        self.voice_name = self.synthesizer_config.voice_name
        self.pitch = self.synthesizer_config.pitch
        self.rate = self.synthesizer_config.rate
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)
        self.logger = logger or logging.getLogger(__name__)

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        async def generate_filler_audio(filler_phrase):
            cache_key = "-".join(
                (
                    str(filler_phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.voice_name),
                    str(self.pitch),
                    str(self.rate),
                )
            )
            filler_audio_path = os.path.join(FILLER_AUDIO_PATH, f"{cache_key}.bytes")
            if not os.path.exists(filler_audio_path):
                self.logger.debug(f"Generating filler audio for {filler_phrase.text}")
                ssml = self.create_ssml(filler_phrase.text, volume=50, rate=4)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor, self.synthesizer.speak_ssml, ssml
                )
                offset = self.synthesizer_config.sampling_rate * self.OFFSET_MS // 1000
                audio_data = result.audio_data[offset:]
                with open(filler_audio_path, "wb") as f:
                    f.write(audio_data)
            else:
                with open(filler_audio_path, "rb") as f:
                    audio_data = f.read()
            return FillerAudio(
                filler_phrase,
                audio_data,
                self.synthesizer_config,
            )

        tasks = [generate_filler_audio(phrase) for phrase in FILLER_PHRASES]
        filler_phrase_audios = await asyncio.gather(*tasks)
        return filler_phrase_audios

    async def get_phrase_affirmative_audios(self) -> List[FillerAudio]:
        async def generate_affirmative_audio(affirmative_phrase):
            cache_key = "-".join(
                (
                    str(affirmative_phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.voice_name),
                    str(self.pitch),
                    str(self.rate),
                )
            )
            affirmative_audio_path = os.path.join(
                AFFIRMATIVE_AUDIO_PATH, f"{cache_key}.bytes"
            )
            if not os.path.exists(affirmative_audio_path):
                self.logger.debug(
                    f"Generating affirmative audio for {affirmative_phrase.text}"
                )
                ssml = self.create_ssml(affirmative_phrase.text, volume=55, rate=2)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor, self.synthesizer.speak_ssml, ssml
                )
                offset = self.synthesizer_config.sampling_rate * self.OFFSET_MS // 1000
                audio_data = result.audio_data[offset:]
                with open(affirmative_audio_path, "wb") as f:
                    f.write(audio_data)
            else:
                with open(affirmative_audio_path, "rb") as f:
                    audio_data = f.read()
            return FillerAudio(
                affirmative_phrase,
                audio_data,
                self.synthesizer_config,
            )

        tasks = [generate_affirmative_audio(phrase) for phrase in AFFIRMATIVE_PHRASES]
        affirmative_phrase_audios = await asyncio.gather(*tasks)
        self.logger.debug("Affirmative audios generated")
        return affirmative_phrase_audios

    def add_marks(self, message: str, index=0) -> str:
        search_result = re.search(r"([\.\,\:\;\-\—]+)", message)
        if search_result is None:
            return message
        start, end = search_result.span()
        with_mark = message[:start] + f'<mark name="{index}" />' + message[start:end]
        rest = message[end:]
        rest_stripped = re.sub(r"^(.+)([\.\,\:\;\-\—]+)$", r"\1", rest)
        if len(rest_stripped) == 0:
            return with_mark
        return with_mark + self.add_marks(rest_stripped, index + 1)

    def word_boundary_cb(self, evt, pool):
        pool.add(evt)

    def create_ssml(
        self,
        message: str,
        bot_sentiment: Optional[BotSentiment] = None,
        volume: int = 1,
        rate: int = 1,
    ) -> str:
        is_neural = "neural" in self.voice_name.lower()
        voice_language_code = (
            self.synthesizer_config.language_code if is_neural else None
        )
        rate = 0.1  # This rate assignment seems to be constant in both conditions

        code = f'"{voice_language_code}"' if voice_language_code else "None"
        if is_neural:
            ssml_root = ElementTree.fromstring(
                f'<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis"{f" xml:lang={code}" if voice_language_code else ""}></speak>'
            )
        else:
            ssml_root = ElementTree.fromstring(
                f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'></speak>"
            )
        voice = ElementTree.SubElement(ssml_root, "voice")
        voice.set("name", self.voice_name if is_neural else "PhoenixLatestNeural")

        if not is_neural:
            speaker = ElementTree.SubElement(
                voice, "{%s}ttsembedding" % NAMESPACES.get("mstts")
            )
            speaker.set("speakerProfileId", self.voice_name)
            speaker.text = message.strip()
        voice_root = voice
        if is_neural and self.synthesizer_config.language_code not in ["en-US", "en"]:
            lang = ElementTree.SubElement(voice, "{%s}lang" % NAMESPACES.get(""))
            lang.set("xml:lang", self.synthesizer_config.language_code)
            voice_root = lang

        if bot_sentiment and bot_sentiment.emotion and is_neural:
            styled = ElementTree.SubElement(
                voice_root, "{%s}express-as" % NAMESPACES.get("mstts")
            )
            styled.set("style", bot_sentiment.emotion)
            styled.set(
                "styledegree", str(bot_sentiment.degree * 2)
            )  # Azure specific, it's a scale of 0-2
            voice_root = styled

        if " " in message and is_neural:
            silence = ElementTree.SubElement(
                voice_root, "{%s}silence" % NAMESPACES.get("mstts")
            )
            silence.set("value", f"{random.randint(80, 130)}ms")
            silence.set("type", "comma-exact")
        if is_neural:
            prosody = ElementTree.SubElement(voice_root, "prosody")
            prosody.set("pitch", f"{self.pitch}%")
            prosody.set("rate", f"{rate*self.rate}%")
            prosody.set("volume", f"-{volume}%")
            prosody.text = message.strip()

        return (
            ElementTree.tostring(ssml_root, encoding="unicode")
            .replace("ns0:", "")
            .replace(":ns0", "")
            .replace("ns0", "")
        )

    def synthesize_ssml(self, ssml: str) -> speechsdk.AudioDataStream:
        result = self.synthesizer.start_speaking_ssml_async(ssml).get()
        return speechsdk.AudioDataStream(result)

    def ready_synthesizer(self):
        connection = speechsdk.Connection.from_speech_synthesizer(self.synthesizer)
        connection.open(True)

    # given the number of seconds the message was allowed to go until, where did we get in the message?
    def get_message_up_to(
        self,
        message: str,
        ssml: str,
        seconds: float,
        word_boundary_event_pool: WordBoundaryEventPool,
    ) -> str:
        events = word_boundary_event_pool.get_events_sorted()
        # for event in events:
        #     if event["audio_offset"] > seconds:
        #         ssml_fragment = ssml[: event["text_offset"]]
        #         # TODO: this is a little hacky, but it works for now
        #         return ssml_fragment.split(">")[-1]
        return message

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        # this is so it says numbers slowly
        def remove_dashes(match):
            return match.group().replace("-", "...").replace("+", "")

        def format_digits(match):
            digits = match.group()
            digits = re.sub(
                r"(\d)-(\d)", r"\1\2", digits
            )  # Remove dashes between two numbers

            if len(digits) <= 7:
                formatted = ""
                for digit in digits:
                    formatted += digit + "... "
                return formatted

            first_digit = ""
            if len(digits) % 2 != 0:
                first_digit = digits[0]
                digits = digits[1:]

            formatted = ""
            for i in range(0, len(digits) - 4, 3):
                formatted += (
                    digits[i] + ", " + digits[i + 1] + ", " + digits[i + 2] + "... "
                )
            formatted += digits[-4:-2] + "... " + digits[-2:]

            if first_digit:
                ret = first_digit + "... " + formatted
                return ret
            return formatted

        modified_message = re.sub(
            r"\b\d+-\d+\b",
            remove_dashes,
            message.text.replace("-", " ").replace(" (", "").replace(") ", ""),
        )

        modified_message = re.sub(r"\b(\d{5,})\b", format_digits, modified_message)

        # offset = int(self.OFFSET_MS * (self.synthesizer_config.sampling_rate / 1000))
        offset = 0
        self.logger.debug(f"Synthesizing message: {message}")

        # Azure will return no audio for certain strings like "-", "[-", and "!"
        # which causes the `chunk_generator` below to hang. Return an empty
        # generator for these cases.
        if not re.search(r"\w", message.text):
            return SynthesisResult(
                self.empty_generator(),
                lambda _: message.text,
            )

        async def chunk_generator(
            audio_data_stream: speechsdk.AudioDataStream, chunk_transform=lambda x: x
        ):
            audio_buffer = bytes(chunk_size)
            filled_size = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor,
                lambda: audio_data_stream.read_data(audio_buffer),
            )
            if filled_size != chunk_size:
                yield SynthesisResult.ChunkResult(
                    chunk_transform(audio_buffer[offset:]), True
                )
                return
            else:
                yield SynthesisResult.ChunkResult(
                    chunk_transform(audio_buffer[offset:]), False
                )
            while True:
                filled_size = audio_data_stream.read_data(audio_buffer)
                if filled_size != chunk_size:
                    yield SynthesisResult.ChunkResult(
                        chunk_transform(audio_buffer[: filled_size - offset]), True
                    )
                    break
                yield SynthesisResult.ChunkResult(chunk_transform(audio_buffer), False)

        word_boundary_event_pool = WordBoundaryEventPool()
        self.synthesizer.synthesis_word_boundary.connect(
            lambda event: self.word_boundary_cb(event, word_boundary_event_pool)
        )
        ssml = (
            message.ssml
            if isinstance(message, SSMLMessage)
            # put modified here so it doesnt mess up transcript but says slowly
            else self.create_ssml(modified_message, bot_sentiment=bot_sentiment)
        )
        audio_data_stream = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool_executor, self.synthesize_ssml, ssml
        )
        if self.synthesizer_config.should_encode_as_wav:
            output_generator = chunk_generator(
                audio_data_stream,
                lambda chunk: encode_as_wav(chunk, self.synthesizer_config),
            )
        else:
            output_generator = chunk_generator(audio_data_stream)

        return SynthesisResult(
            output_generator,
            lambda seconds: self.get_message_up_to(
                message.text, ssml, seconds, word_boundary_event_pool
            ),
        )
