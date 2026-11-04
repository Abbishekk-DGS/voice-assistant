"""Microphone capture with automatic endpointing, plus audio file helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what Whisper expects


@dataclass
class Endpointer:
    """Energy-based voice activity endpointing with an adaptive noise floor.

    Speech starts when frame energy rises `start_ratio` above the measured noise floor and
    ends after `silence_ms` of quiet. Simple, CPU-free, and good enough for a desk mic."""

    frame_ms: int = 30
    start_ratio: float = 3.0
    stop_ratio: float = 1.8
    silence_ms: int = 800
    max_seconds: float = 20.0
    calibration_ms: int = 300

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.noise: list[float] = []
        self.floor: float | None = None
        self.speaking = False
        self.silent_frames = 0
        self.frames: list[np.ndarray] = []

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)

    def push(self, frame: np.ndarray) -> bool:
        """Feed one frame. Returns True when an utterance has ended."""
        energy = self.rms(frame)
        calib_frames = self.calibration_ms // self.frame_ms
        if self.floor is None:
            self.noise.append(energy)
            if len(self.noise) >= calib_frames:
                self.floor = max(float(np.median(self.noise)), 1e-4)
            return False
        if not self.speaking:
            if energy > self.floor * self.start_ratio:
                self.speaking = True
                self.frames.append(frame)
            else:  # keep adapting to the room while waiting
                self.floor = 0.95 * self.floor + 0.05 * energy
            return False
        self.frames.append(frame)
        self.silent_frames = self.silent_frames + 1 if energy < self.floor * self.stop_ratio else 0
        too_long = len(self.frames) * self.frame_ms / 1000 >= self.max_seconds
        return self.silent_frames * self.frame_ms >= self.silence_ms or too_long

    def audio(self) -> np.ndarray:
        return np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)


def listen(endpointer: Endpointer | None = None, device=None) -> np.ndarray:
    """Block until the user says something; return 16 kHz mono float32 audio."""
    import queue

    import sounddevice as sd

    ep = endpointer or Endpointer()
    ep.reset()
    q: queue.Queue = queue.Queue()
    blocksize = SAMPLE_RATE * ep.frame_ms // 1000
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=blocksize, device=device,
        callback=lambda data, *_: q.put(data[:, 0].copy()),
    ):  # fmt: skip
        while not ep.push(q.get()):
            pass
    return ep.audio()


def load_audio(path: Path) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def play(audio: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(audio, sample_rate)
    sd.wait()
