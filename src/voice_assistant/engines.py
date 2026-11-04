"""Speech-to-text, the LLM (with tool calling) and text-to-speech."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np

from .tools import SCHEMAS, run_tool

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"
CACHE = Path(os.environ.get("VOICE_ASSISTANT_CACHE", Path.home() / ".cache" / "voice-assistant"))

SYSTEM_PROMPT = """You are a helpful voice assistant running entirely on the user's computer.
Your replies are spoken aloud, so:
- Answer in one to three short sentences unless asked for more.
- No markdown, lists, emoji, URLs or code. Write numbers the way you'd say them.
- Answer in English, using only the Latin alphabet: never Chinese characters or other scripts,
  because the voice can only pronounce English.
- Use the calculate tool for any arithmetic and current_time for the date or time."""


class WhisperSTT:
    def __init__(self, model: str = "base.en", language: str | None = None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="auto", compute_type="int8")
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class OllamaChat:
    """Streaming chat that transparently runs tool calls before the spoken answer."""

    def __init__(self, model: str = "qwen2.5:3b", url: str = OLLAMA_URL, max_history: int = 12):
        self.model, self.url, self.max_history = model, url, max_history
        self.history: list[dict] = []
        self.tool_log: list[tuple[str, dict, str]] = []

    def _stream(self, messages: list[dict], tools: bool) -> Iterator[dict]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.4},
        }
        if tools:
            payload["tools"] = SCHEMAS
        with httpx.stream("POST", f"{self.url}/api/chat", json=payload, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line)

    def reply(self, user_text: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": user_text})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-self.max_history :],
        ]
        answer = ""
        for _ in range(3):  # at most 2 rounds of tool use before answering
            calls = []
            for event in self._stream(messages, tools=True):
                msg = event.get("message", {})
                calls += msg.get("tool_calls") or []
                if token := msg.get("content"):
                    answer += token
                    yield token
            if not calls:
                break
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for call in calls:
                fn = call["function"]
                result = run_tool(fn["name"], fn.get("arguments") or {})
                self.tool_log.append((fn["name"], fn.get("arguments") or {}, result))
                messages.append({"role": "tool", "content": result, "tool_name": fn["name"]})
        self.history.append({"role": "assistant", "content": answer})


class PiperTTS:
    def __init__(self, voice: str = "en_US-lessac-medium"):
        from piper import PiperVoice
        from piper.download_voices import download_voice

        CACHE.mkdir(parents=True, exist_ok=True)
        model = CACHE / f"{voice}.onnx"
        if not model.exists():
            download_voice(voice, CACHE)
        self.voice = PiperVoice.load(model)
        self.sample_rate = self.voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
