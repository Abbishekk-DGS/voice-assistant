"""voice-assistant: talk to a local LLM. Nothing leaves your machine.

voice-assistant                          # hands-free conversation with the microphone
voice-assistant --input question.wav     # one turn from a file, reply saved as a WAV
voice-assistant --text "what's 23 times 17?" --save reply.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voice-assistant",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama chat model")
    parser.add_argument("--whisper", default="base.en", help="faster-whisper model size")
    parser.add_argument("--voice", default="en_US-lessac-medium", help="Piper voice name")
    parser.add_argument("--input", help="audio file to answer (skips the microphone)")
    parser.add_argument("--text", help="typed question (skips speech recognition)")
    parser.add_argument("--save", help="write the spoken reply to this WAV file")
    parser.add_argument("--mute", action="store_true", help="don't play audio")
    parser.add_argument("--stats", action="store_true", help="print latency for every turn")
    args = parser.parse_args(argv)

    from .assistant import Assistant
    from .audio import Endpointer, listen, load_audio, play
    from .engines import OllamaChat, PiperTTS, WhisperSTT

    print("loading models...")
    stt = None if args.text else WhisperSTT(args.whisper)
    tts = PiperTTS(args.voice)
    bot = Assistant(stt, OllamaChat(args.model), tts, player=None if args.mute else play)

    def show(stats) -> None:
        if args.stats:
            print(
                f"   [stt {stats.stt_ms:.0f} ms · first token {stats.first_token_ms:.0f} ms · "
                f"first audio {stats.first_audio_ms:.0f} ms · total {stats.total_ms:.0f} ms]"
            )
        for name, arguments, result in bot.chat.tool_log:
            print(f"   (tool {name}({arguments}) -> {result})")
        bot.chat.tool_log.clear()

    def speak_line(sentence: str) -> None:
        print(f"assistant: {sentence}")

    if args.input or args.text:
        audio = load_audio(Path(args.input)) if args.input else None
        stats = bot.turn(audio=audio, text=args.text, on_text=speak_line)
        if args.input:
            print(f"you: {stats.transcript}")
        show(stats)
        if args.save and stats.audio:
            import soundfile as sf

            sf.write(args.save, np.concatenate(stats.audio), tts.sample_rate)
            print(f"saved {args.save}")
        return 0

    endpointer = Endpointer()
    print("Listening. Speak naturally; pause to finish. Ctrl-C to quit.")
    try:
        while True:
            audio = listen(endpointer)
            if len(audio) < 16_000 * 0.3:
                continue
            stats = bot.turn(audio=audio, on_text=speak_line)
            if stats.transcript:
                print(f"you: {stats.transcript}")
                show(stats)
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
