"""
audio_capture.py -- captures a participant's spoken answer from Reachy
Mini's microphone, for the ask_when_ambiguous condition's clarifying
question. Used instead of the facilitator typing the answer in
(trial_runner.console_get_answer is now only a safety-net fallback).

Verified directly against the installed reachy_mini==1.12.0 SDK source
(media/audio_base.py, media/webrtc_client_gstreamer.py), not guessed:

  - For connection_mode="network" (what this project uses), audio
    capture is ALWAYS ON once connected -- start_recording()/
    stop_recording() are literal no-ops for this backend.
  - mini.media.get_audio_sample() pulls ONE buffered chunk at a time from
    a bounded GStreamer queue (drop=True, max-buffers=500), returning a
    float32 array of shape (num_samples, 2) (stereo), or None if nothing
    is queued yet. It does NOT return the whole recording in one call --
    you have to poll it in a loop and concatenate chunks yourself. (An
    earlier quick test that did start_recording() -> sleep(3) -> one
    get_audio_sample() call only ever captured ONE chunk, not the full
    3 seconds -- that's why this file exists.)
  - AudioBase.SAMPLE_RATE = 16000, which happens to be exactly what
    faster-whisper expects, so no resampling step is needed.

KNOWN RISK, not a software problem: if Reachy's mic picks up its own
speaker playing the clarifying question (acoustic leakage/echo), the
transcription can come back polluted with the robot's own words. The
drain step below clears out whatever was buffered *before* we start
really listening, but it can't fix ongoing echo during the recording
itself. Test this for real against your room/robot setup -- if it's a
problem, the fix is a bigger pause after say() finishes before
record_answer() starts, not more code here.
"""
import time

import numpy as np

SAMPLE_RATE = 16000


def _drain(mini, max_drain_seconds: float = 0.5):
    """Always-on audio means stale chunks (ambient noise from the
    between-trial wait, or playback tail) can already be queued before
    we actually want to start listening. Pull and discard whatever's
    sitting there so the real recording window starts clean."""
    t0 = time.time()
    while time.time() - t0 < max_drain_seconds:
        if mini.media.get_audio_sample() is None:
            break


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def record_answer(
    mini,
    max_seconds: float = 6.0,
    min_seconds: float = 0.6,
    silence_rms: float = 0.01,
    silence_hang_seconds: float = 1.0,
    poll_interval: float = 0.05,
):
    """Records until EITHER max_seconds elapses, or the participant has
    spoken (RMS over silence_rms at least once, after min_seconds has
    passed) and then gone quiet for silence_hang_seconds. Returns mono
    float32 samples at 16kHz, or None if nothing was ever captured.

    silence_rms is a guess -- test this against your actual room/mic
    before a real session (run this file directly, see below) and raise
    it if it cuts people off mid-sentence, lower it if it never detects
    silence and always runs to max_seconds.
    """
    _drain(mini)

    chunks = []
    heard_speech = False
    silence_started = None
    t0 = time.time()

    while time.time() - t0 < max_seconds:
        sample = mini.media.get_audio_sample()
        if sample is None:
            time.sleep(poll_interval)
            continue

        mono = sample.mean(axis=1).astype(np.float32)  # stereo -> mono
        chunks.append(mono)
        level = _rms(mono)
        elapsed = time.time() - t0

        if level >= silence_rms:
            heard_speech = True
            silence_started = None
        elif heard_speech and elapsed >= min_seconds:
            if silence_started is None:
                silence_started = time.time()
            elif time.time() - silence_started >= silence_hang_seconds:
                break

    if not chunks:
        return None
    return np.concatenate(chunks)


if __name__ == "__main__":
    # Quick standalone test against the real robot, run this BEFORE your
    # first real session to sanity-check the mic + tune silence_rms:
    #   python audio_capture.py
    # It prints "Listening..." -- say something short right after, like
    # "the blue one", then check what comes back.
    from reachy_mini import ReachyMini
    from config import ROBOT_IP
    from stt import transcribe

    with ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend="default") as mini:
        mini.media.start_playing()
        time.sleep(2.0)
        print("Listening... say something short, like 'the blue one'.")
        samples = record_answer(mini)
        if samples is None:
            print("Got nothing -- no audio was buffered at all.")
        else:
            print(f"Recorded {samples.shape[0] / SAMPLE_RATE:.2f}s of audio.")
            text = transcribe(samples)
            print(f"Transcribed: {text!r}")
        mini.media.stop_playing()