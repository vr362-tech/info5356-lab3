"""
robot_actions.py -- the robot's physical side: grab a camera frame, turn
the head to point at a pixel location, and speak. All calls are verified
directly against the installed reachy_mini==1.12.0 SDK source (not guessed):

  - mini.media.get_frame_jpeg()      -> bytes, JPEG-encoded current frame
  - mini.look_at_image(u, v, duration=...) -> moves the head to look at
    pixel (u, v) in that same camera frame (counted from the top-left,
    u = across, v = down).
    IMPORTANT (verified against reachy_mini.py:look_at_image and
    vision/look_at.py:look_at_image_pose): this computes the target angle
    using T_world_head=self.get_current_head_pose() -- the head's CURRENT
    pose at the moment you call it, NOT the pose the frame was captured
    at. So point_at() must be called while the head is still physically
    at the same position it was in when that (u, v) was found. Moving
    the head (even go_neutral()) in between will make it point at the
    wrong spot.
  - mini.media.play_sound(path)      -> plays a wav/aiff file on the
    robot's speaker
  - mini.goto_target(head=..., antennas=..., duration=...) -> moves to an
    explicit head pose, used for the neutral reset and every scan step.

If your installed SDK version differs and one of these doesn't exist,
run `python -c "from reachy_mini import ReachyMini; help(ReachyMini.look_at_image)"`
in your own terminal (with the venv active) to check the current signature.
"""
import io
import time

import numpy as np
from reachy_mini.utils import create_head_pose

from speech import generate_speech_file


def capture_scene_jpeg(mini) -> bytes:
    jpeg_bytes = mini.media.get_frame_jpeg()
    if jpeg_bytes is None:
        raise RuntimeError(
            "Camera returned no frame -- check the robot is connected and "
            "give the camera a couple seconds to warm up after connecting."
        )
    return jpeg_bytes


def get_jpeg_size(jpeg_bytes: bytes) -> tuple:
    """(width, height) of a captured frame -- used to turn a chosen pixel
    into a left/center/right phrase. Local PIL import so it's only needed
    if you actually use the location-description feature."""
    from PIL import Image
    return Image.open(io.BytesIO(jpeg_bytes)).size


def describe_location(pixel_center, frame_width: int) -> str:
    """Turns a chosen object's pixel x-coordinate into a coarse spoken
    direction, by thirds of the frame width. Deliberately coarse -- this
    is for a one-phrase spoken description, not precise localization."""
    u, _v = pixel_center
    if u < frame_width / 3:
        return "on the left"
    if u > 2 * frame_width / 3:
        return "on the right"
    return "in the center"


def go_neutral(mini):
    neutral = create_head_pose(yaw=0, pitch=0, roll=0, degrees=True)
    mini.goto_target(head=neutral, antennas=np.array([0.0, 0.0]), duration=0.5)
    time.sleep(1.2)  # let the head fully settle before the next frame grab


def goto_pose(mini, yaw: float, pitch: float):
    """Moves to an explicit (yaw, pitch) scan position -- same movement
    locate_object() uses for each sweep step, exposed separately so
    run_one_trial can return the head to the exact pose a chosen object
    was found at before pointing (see look_at_image's note above)."""
    pose = create_head_pose(yaw=yaw, pitch=pitch, roll=0, degrees=True)
    mini.goto_target(head=pose, antennas=np.array([0.0, 0.0]), duration=0.6)
    time.sleep(1.0)


def point_at(mini, pixel_center, duration: float = 1.0):
    """Call this ONLY while the head is still at the pose the frame
    containing pixel_center was captured at -- see the module docstring
    and goto_pose(). Do not move the head (go_neutral included) between
    finding the object and calling this."""
    u, v = pixel_center
    mini.look_at_image(int(u), int(v), duration=duration)
    time.sleep(duration)


def say(mini, text: str):
    wav_path = generate_speech_file(text)
    mini.media.play_sound(str(wav_path))
    # play_sound() returns before playback finishes -- wait out a rough
    # estimate of the audio's duration so the next action (or disconnect,
    # or an audio recording that starts right after) doesn't cut the
    # robot off mid-sentence or capture its own tail end.
    estimated_duration = max(1.5, len(text.split()) * 0.45 + 0.5)
    time.sleep(estimated_duration)


# (yaw, pitch) degrees: center, then left, then right. Center first since
# it's the natural "look at the table" starting point; left/right are a
# fixed sweep, not model-chosen (mirrors the course's own VLA demo's
# search behavior, scoped down to 3 fixed positions). This now runs on
# EVERY trial, not just as a last-resort fallback -- Reachy visibly looks
# around before deciding, per the study's intended behavior.
SCAN_POSITIONS = [(0, 0), (-20, 0), (20, 0)]


def locate_object(mini, request_text: str):
    """Always sweeps center -> left -> right, capturing and describing a
    fresh frame at each head position. All objects seen across the three
    views are merged into one list (first sighting wins if the same
    label shows up twice) before asking the LLM whether the request is
    ambiguous -- so the ambiguity decision reflects everything Reachy can
    see from this vantage sweep, not just whichever position went first.

    Each merged object remembers which (yaw, pitch) position it was seen
    from, under "_pose", so the caller can return the head to that exact
    pose before point_at() -- required for look_at_image's geometry, see
    the module docstring.

    Returns (frames, merged_objects, interpretation):
      frames          -- {(yaw, pitch): jpeg_bytes} for all 3 positions
      merged_objects  -- [{"label", "pixel_center", "_pose": (yaw, pitch)}, ...]
      interpretation  -- interpret_request(request_text, merged_objects)
    """
    from scene_vlm import describe_scene
    from request_interpreter import interpret_request

    frames = {}
    merged_objects = []
    seen_labels = set()

    for yaw, pitch in SCAN_POSITIONS:
        goto_pose(mini, yaw, pitch)
        jpeg_bytes = capture_scene_jpeg(mini)
        frames[(yaw, pitch)] = jpeg_bytes
        for obj in describe_scene(jpeg_bytes):
            key = obj["label"].strip().lower()
            if key in seen_labels:
                continue  # already seen from an earlier position
            seen_labels.add(key)
            merged_objects.append({**obj, "_pose": (yaw, pitch)})

    interpretation = interpret_request(request_text, merged_objects)
    return frames, merged_objects, interpretation