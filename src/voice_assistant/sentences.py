"""Turn a token stream into speakable sentences as soon as each one is complete.

This is where the latency win comes from: the first sentence goes to TTS while the LLM is
still writing the second, so the assistant starts talking after one sentence, not after the
whole reply.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
    "approx",
    "no",
}
_BOUNDARY = re.compile(r"([.!?]+)(['\")\]]*)(\s+)")


def _is_real_boundary(text: str, end: int) -> bool:
    """A '.' ends a sentence unless it follows an abbreviation, an initial, or sits in a number."""
    before = text[:end].rstrip(".!?'\")]")
    word = before.rsplit(None, 1)[-1].lower() if before.split() else ""
    if word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
        return False
    return True


def clean_for_speech(text: str) -> str:
    """Strip markdown that TTS would read aloud literally."""
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"[*_#>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return " ".join(text.split())


def stream_sentences(tokens: Iterable[str], min_chars: int = 12) -> Iterator[str]:
    buffer = ""
    for token in tokens:
        buffer += token
        while True:
            cut = None
            for m in _BOUNDARY.finditer(buffer):
                if m.end() >= min_chars and _is_real_boundary(buffer, m.start()):
                    cut = m.end()
                    break
            if cut is None:
                # a newline after enough text is also a natural pause (lists, paragraphs)
                nl = buffer.find("\n", min_chars)
                if nl == -1:
                    break
                cut = nl + 1
            sentence = clean_for_speech(buffer[:cut])
            buffer = buffer[cut:]
            if sentence:
                yield sentence
    tail = clean_for_speech(buffer)
    if tail:
        yield tail
