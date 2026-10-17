"""
speech.py -- text-to-speech via macOS's built-in `say`, converted to a wav
file for the robot's play_sound(). No API key, no extra install.

Why not pre-recorded clips like Lab 2? Because the clarifying questions
and object names here are only known at runtime (they come out of the
LLM), so there's no fixed set of clips to record in advance.

If a teammate runs this on Linux/Windows instead of a Mac, swap
generate_speech_file()'s body for `espeak` (Linux) or any TTS API --
everything else in this file and in robot_actions.py stays the same.
"""
import subprocess
import tempfile
from pathlib import Path


def generate_speech_file(text: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    aiff_path = tmp_dir / "speech.aiff"
    wav_path = tmp_dir / "speech.wav"

    subprocess.run(["say", "-o", str(aiff_path), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(wav_path)],
        check=True,
    )
    return wav_path


if __name__ == "__main__":
    # Quick test: python speech.py "hello from the robot"
    import sys
    path = generate_speech_file(" ".join(sys.argv[1:]) or "testing one two three")
    print(f"Wrote {path}")
