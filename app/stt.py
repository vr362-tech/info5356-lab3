"""
stt.py -- transcribes a participant's recorded answer (from
audio_capture.record_answer) using faster-whisper, running locally (no
cloud API key needed). Loaded lazily so dev/offline (--image) runs that
never touch the robot's mic don't pay the model-load cost.

Install first (on your Mac, inside the venv):
    pip install faster-whisper

First real use downloads and caches the model from Hugging Face -- run
audio_capture.py once on its own, ahead of your real session, so trial 1
isn't slowed down by a model download mid-experiment.

Verified against the installed faster-whisper package in this sandbox
(inspect.signature on WhisperModel.transcribe): it accepts a numpy array
directly for `audio`, and when you pass one, it is NOT resampled -- it's
used as-is, so it must already be mono float32 at the model's expected
sampling rate (16000 Hz, same as feature_extractor.sampling_rate). That
matches reachy_mini's AudioBase.SAMPLE_RATE exactly, so no resampling
step is needed between audio_capture.py and here.
"""
from faster_whisper import WhisperModel

_model = None


def _get_model():
    global _model
    if _model is None:
        # "base.en" is a good speed/accuracy tradeoff on CPU (Apple
        # Silicon included -- faster-whisper runs on CTranslate2, not
        # GPU-only torch) for short, English answers like "the blue
        # one". Swap to "small.en" if accuracy is poor and you have the
        # time budget; "tiny.en" if trials are timing out.
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_samples, sample_rate: int = 16000) -> str:
    """audio_samples: mono float32 numpy array at `sample_rate` Hz
    (audio_capture.record_answer's output), or None. Returns the
    transcribed text, stripped, or "" if there's nothing to transcribe
    or nothing intelligible came back.
    """
    if audio_samples is None or audio_samples.size == 0:
        return ""
    if sample_rate != 16000:
        raise ValueError(
            f"faster-whisper expects 16kHz audio when given a numpy array "
            f"directly (no resampling happens); got {sample_rate}Hz."
        )
    segments, _info = _get_model().transcribe(audio_samples, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


if __name__ == "__main__":
    # Quick standalone test with no robot needed, using a WAV file:
    #   python stt.py path/to/some_clip.wav
    import sys
    import soundfile as sf

    data, sr = sf.read(sys.argv[1], dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        raise SystemExit(
            f"{sys.argv[1]} is {sr}Hz, not 16000Hz -- resample it first "
            f"(this quick test doesn't resample, transcribe() doesn't either)."
        )
    print(repr(transcribe(data)))