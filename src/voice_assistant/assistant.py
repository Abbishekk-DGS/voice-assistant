"""The conversational loop: hear -> think -> speak, with the stages overlapped.

LLM thread:   token token token . | token token . | ...
                          ↓ sentence        ↓ sentence
speech thread:       synthesize → play   synthesize → play
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .sentences import stream_sentences


@dataclass
class TurnStats:
    transcript: str = ""
    reply: str = ""
    stt_ms: float = 0.0
    first_token_ms: float = 0.0
    first_audio_ms: float = 0.0  # end of user speech -> first sound out
    total_ms: float = 0.0
    sentences: int = 0
    audio: list[np.ndarray] = field(default_factory=list)


class Assistant:
    def __init__(self, stt, chat, tts, player: Callable[[np.ndarray, int], None] | None = None):
        self.stt, self.chat, self.tts = stt, chat, tts
        self.player = player

    def turn(
        self, audio: np.ndarray | None = None, text: str | None = None, on_text=None
    ) -> TurnStats:
        stats = TurnStats()
        t0 = time.perf_counter()
        if text is None:
            text = self.stt.transcribe(audio)
        stats.transcript = text
        stats.stt_ms = (time.perf_counter() - t0) * 1000
        if not text:
            return stats

        sentences: queue.Queue[str | None] = queue.Queue()
        first_token: list[float] = []

        def produce() -> None:
            def timed_tokens():
                for tok in self.chat.reply(text):
                    if not first_token:
                        first_token.append(time.perf_counter())
                    stats.reply += tok
                    yield tok

            try:
                for sentence in stream_sentences(timed_tokens()):
                    sentences.put(sentence)
            finally:
                sentences.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        while (sentence := sentences.get()) is not None:
            if on_text:
                on_text(sentence)
            wav = self.tts.synthesize(sentence)
            if not stats.first_audio_ms:
                stats.first_audio_ms = (time.perf_counter() - t0) * 1000
            stats.sentences += 1
            stats.audio.append(wav)
            if self.player:
                self.player(wav, self.tts.sample_rate)
        producer.join()
        if first_token:
            stats.first_token_ms = (first_token[0] - t0) * 1000
        stats.total_ms = (time.perf_counter() - t0) * 1000
        return stats
