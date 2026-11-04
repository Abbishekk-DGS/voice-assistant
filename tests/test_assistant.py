import time

import numpy as np

from voice_assistant.assistant import Assistant
from voice_assistant.audio import Endpointer
from voice_assistant.sentences import clean_for_speech, stream_sentences
from voice_assistant.tools import calculate, run_tool


def tokens(text, size=3):
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_sentences_stream_out_early_and_respect_abbreviations():
    text = "Dr. Rao arrived at 3.5 p.m. today. It was raining! Was it cold? Yes"
    out = list(stream_sentences(tokens(text)))
    assert out == ["Dr. Rao arrived at 3.5 p.m. today.", "It was raining!", "Was it cold?", "Yes"]


def test_markdown_is_not_read_aloud():
    assert (
        clean_for_speech("**Sure!** Run `ls` and see [docs](http://x)") == "Sure! Run and see docs"
    )


def test_calculator_is_safe_and_exact():
    assert calculate("23*17") == "391"
    assert calculate("2^10") == "1024"
    assert calculate("sqrt(2)") == "1.41421"
    assert calculate("__import__('os').system('rm -rf /')").startswith("error")
    assert calculate("9**9**9").startswith("error")
    assert run_tool("nope", {}).startswith("error")


def test_endpointer_detects_an_utterance():
    rng = np.random.default_rng(0)
    frame = 480  # 30 ms at 16 kHz
    noise = lambda n: rng.normal(0, 0.005, n * frame).astype(np.float32)  # noqa: E731
    speech = (0.2 * np.sin(np.linspace(0, 400 * np.pi, 40 * frame))).astype(np.float32)
    stream = np.concatenate([noise(20), speech, noise(40)])
    ep = Endpointer()
    ended_at = None
    for i in range(0, len(stream), frame):
        if ep.push(stream[i : i + frame]):
            ended_at = i // frame
            break
    assert ended_at is not None and 20 + 40 < ended_at < 20 + 40 + 35
    assert len(ep.audio()) >= len(speech)


class SlowChat:
    """Emits two sentences with a pause between them."""

    def reply(self, text):
        for tok in tokens("Sure, here is the first sentence. "):
            yield tok
        time.sleep(0.3)
        for tok in tokens("And here is the second one."):
            yield tok


class FakeTTS:
    sample_rate = 22050

    def __init__(self):
        self.spoken = []

    def synthesize(self, text):
        self.spoken.append((text, time.perf_counter()))
        return np.zeros(100, np.float32)


def test_first_sentence_is_spoken_before_the_reply_finishes():
    tts = FakeTTS()
    stats = Assistant(stt=None, chat=SlowChat(), tts=tts).turn(text="hello")
    assert [s for s, _ in tts.spoken] == [
        "Sure, here is the first sentence.",
        "And here is the second one.",
    ]
    assert tts.spoken[1][1] - tts.spoken[0][1] > 0.25  # first one went out during the pause
    assert stats.first_audio_ms < stats.total_ms - 250
    assert stats.reply.endswith("second one.")
