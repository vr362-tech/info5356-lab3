"""
trial_runner.py -- runs one spoken request through the full pipeline:
  look around (center -> left -> right) -> describe (VLM) each view ->
  merge what was seen -> interpret (LLM) -> policy (always_act or
  ask_when_ambiguous) -> point the head + say the object name and where
  it is -> log everything to logs/trial_log.csv.

Logs a `success` column (chosen_label vs. the predetermined correct_object
from requests.csv) and a `participant_id` for the within-subject design.
`console_wait_seconds` is tracked separately from the rest of the trial's
elapsed time -- for ask_when_ambiguous trials this is now the time spent
recording + transcribing the participant's spoken answer (see
audio_capture.py/stt.py), not facilitator typing time, except on the
fallback path below.
`failure_reason` is left blank for a human to fill in after reviewing what
actually happened (ASR / VLM perception / ambiguity classification / LLM
target selection / clarification answer interpretation / robot motion /
other) -- it isn't something the code can judge on its own.

Also saves exactly one JPEG per logged trial to logs/images/ (see
save_trial_image), with its path in the `image_path` column -- the scan
frame that actually contained the chosen object, the center-view frame on
a "nothing matched" trial, or the static test image on --image dev runs.
One image per row, always, so logs/images/ and trial_log.csv stay 1:1.

ON THE LIVE ROBOT, EVERY TRIAL:
  1. terminal shows what the participant should say, waits for you to
     press Enter once they've said it;
  2. Reachy always looks around first -- sweeps center -> left -> right,
     capturing and describing a fresh frame at each position (see
     robot_actions.locate_object) -- then decides: acts immediately
     (always_act) or speaks its clarifying question out loud and listens
     for the participant's spoken answer via the real microphone + local
     Whisper transcription (ask_when_ambiguous -- see audio_capture.py/
     stt.py), then points at the chosen object and says its name and a
     coarse location ("deep bowl, on the left");
  3. terminal waits for you to press Enter again before moving to the
     next trial.
This pacing, and the always-on scan, only apply to --robot runs; --image
dev/debug runs stay fast and automatic since there's no participant or
physical robot to pace for.

REAL PARTICIPANT SESSIONS -- the official 3-condition design
--------------------------------------------------------------
Per the experiment guide: C1 (Always Act + Clear), C2 (Always Act +
Ambiguous), C3 (Ask When Ambiguous + Ambiguous) -- "Ask When Ambiguous +
Clear" is intentionally not tested. C1 always runs first as the baseline;
C2/C3 order then alternates by participant (odd-numbered participants get
C2 before C3, even-numbered get C3 before C2). requests.csv needs a
`condition` column (C1/C2/C3) alongside request_text/label/mode/correct_object
-- mode is read per-row, so you don't pass --mode for a session run:

    python trial_runner.py --session ../requests.csv --robot --participant P01

FOUR OTHER WAYS TO RUN IT
---------------------------
1. Dev/offline, no robot needed:
     python trial_runner.py --image ../test_images/table.jpg \\
         --request "look at the cup" --mode ask_when_ambiguous --label ambiguous \\
         --correct-object "red cup" --participant P1

2. One live trial on the robot:
     python trial_runner.py --robot --request "look at the cup" \\
         --mode ask_when_ambiguous --label ambiguous --correct-object "red cup" --participant P1

3. An ad-hoc single-mode batch (dev/debug use -- doesn't need a `condition`
   column, just runs every row under one --mode):
     python trial_runner.py --batch ../requests.csv --mode always_act --robot --participant P1

4. A full --session run against a static test image instead of the robot
   (dev/debug use, no pacing prompts, no scanning -- a static image can't
   be looked around):
     python trial_runner.py --session ../requests.csv --image ../test_images/table.jpg --participant P01
"""
import argparse
import csv
import datetime
import re
import time

from config import LOG_DIR
from scene_vlm import describe_scene
from request_interpreter import interpret_request
from policy import run_always_act, run_ask_when_ambiguous
from robot_actions import (
    go_neutral, locate_object, goto_pose,
    get_jpeg_size, describe_location,
)

LOG_PATH = LOG_DIR / "trial_log.csv"
FIELDNAMES = [
    "participant_id", "condition", "trial_id", "request_text", "label", "mode", "candidates",
    "is_ambiguous_model", "asked_question", "question_text", "answer_text",
    "chosen_label", "correct_object", "success", "start_time", "end_time",
    "elapsed_seconds", "console_wait_seconds", "extra_turns", "failure_reason", "error",
    "image_path",
]


def write_log_row(row: dict):
    LOG_DIR.mkdir(exist_ok=True)
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_trial_image(jpeg_bytes, participant_id, trial_id, start_time) -> str:
    """Saves the one frame that was decisive for this trial -- the scan
    view that actually contained the chosen object if one was found, the
    center-view frame if the trial ended in "nothing matched", or (on
    --image dev/debug runs) the single static test image -- so
    trial_log.csv and logs/images/ stay 1:1: exactly one image per logged
    trial, matching the row count exactly, never zero and never more than
    one. Returns the saved path as a string, or "" if there was no frame
    at all to save (e.g. an exception happened before any frame was
    captured).

    start_time is folded into the filename, not just participant_id +
    trial_id, so re-running the same manual trial_id twice -- which does
    happen during dev/debug -- doesn't silently overwrite an earlier
    run's image out from under its own already-written log row.
    """
    if jpeg_bytes is None:
        return ""
    images_dir = LOG_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_participant = re.sub(r"[^A-Za-z0-9_-]", "_", str(participant_id or "unknown"))
    safe_trial_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(trial_id))
    safe_time = re.sub(r"[^A-Za-z0-9_-]", "_", str(start_time))
    path = images_dir / f"{safe_participant}_{safe_trial_id}_{safe_time}.jpg"
    path.write_bytes(jpeg_bytes)
    return str(path)


def compute_success(chosen_label, correct_object):
    """1 if every word of the ground-truth target appears in what the robot
    chose, 0 if not, "" if there's no ground truth to score against (should
    not happen once requests.csv has a correct_object on every row).
    Word-based, not exact-string, because the VLM's own wording varies
    trial to trial (e.g. "water bottle" vs "plastic water bottle") --
    this only requires the distinguishing words (e.g. "red") to be present.
    """
    if not correct_object or not chosen_label:
        return ""
    target_words = set(correct_object.lower().split())
    chosen_words = set(chosen_label.lower().split())
    return "1" if target_words.issubset(chosen_words) else "0"


def console_get_answer(question_text: str, timing_sink: dict) -> str:
    """Facilitator-typed fallback -- used for dev/offline (--image) runs
    where there's no real robot mic, and as a safety net on the live
    robot if the real speech-to-text capture (see ask_aloud in
    run_one_trial) comes back with an empty transcription. Records how
    long the facilitator took to type the answer into timing_sink.
    """
    print(f"\n[ROBOT ASKS]: {question_text}")
    t0 = time.time()
    answer = input("[Type what the participant said]: ")
    timing_sink["console_wait_seconds"] = round(time.time() - t0, 2)
    return answer


def run_one_trial(jpeg_bytes, request_text, label, mode, correct_object,
                   trial_id, participant_id="", condition="", mini=None):
    start_time = datetime.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    error = ""
    outcome = None
    objects = []
    interpretation = {}
    timing_sink = {}
    frames = None  # {(yaw, pitch): jpeg_bytes}, only set on the live robot
    image_bytes_to_save = None  # whichever single frame ends up "decisive" -- see save_trial_image

    try:
        if mini is not None:
            # Always looks around first (center -> left -> right), every
            # trial -- not just as a fallback. objects below is the
            # merged multi-view list; each object remembers which
            # (yaw, pitch) position it was seen from under "_pose".
            frames, objects, interpretation = locate_object(mini, request_text)
        else:
            objects = describe_scene(jpeg_bytes)
            interpretation = interpret_request(request_text, objects)
            # No robot, no scan -- the single static test image is the
            # only frame there is, so it's what gets logged for this trial.
            image_bytes_to_save = jpeg_bytes

        # Debug visibility into the two LLM steps -- printed on every
        # trial (robot or offline) so you can see exactly what the scene
        # model detected and how the interpreter ranked/chose among it,
        # without having to re-run anything separately. Cheap: just a
        # couple of short lines, no extra API calls.
        print(f"[scene] detected objects: {[o.get('label') for o in objects]}")
        print(
            f"[interpret] candidates={interpretation.get('candidates')} "
            f"is_ambiguous={interpretation.get('is_ambiguous')}"
        )

        def ask_aloud(question_text):
            if mini is not None:
                from robot_actions import say
                say(mini, question_text)
                from audio_capture import record_answer
                from stt import transcribe
                t_listen = time.time()
                samples = record_answer(mini)
                answer = transcribe(samples)
                timing_sink["console_wait_seconds"] = round(time.time() - t_listen, 2)
                if answer:
                    print(f"[STT heard]: {answer!r}")
                    return answer
                print("[STT] heard nothing usable -- falling back to typed input")
            return console_get_answer(question_text, timing_sink)

        if mode == "always_act":
            outcome = run_always_act(interpretation, objects)
        elif mode == "ask_when_ambiguous":
            outcome = run_ask_when_ambiguous(interpretation, objects, get_answer=ask_aloud)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if mini is not None:
            from robot_actions import say

            if outcome.chosen_pixel_center is not None:
                from robot_actions import point_at

                # Find which scan position the chosen object was actually
                # seen from (pixel_center is copied verbatim from `objects`
                # by policy.py, so matching on it recovers the right entry
                # without needing to touch policy.py's simple, deliberately
                # un-fancy logic).
                chosen_obj = next(
                    (o for o in objects if o.get("pixel_center") == outcome.chosen_pixel_center),
                    None,
                )
                chosen_pose = chosen_obj.get("_pose", (0, 0)) if chosen_obj else (0, 0)
                chosen_frame = frames.get(chosen_pose) if frames else jpeg_bytes
                image_bytes_to_save = chosen_frame  # the view that led to this decision

                # IMPORTANT: go to the exact pose the object was found at
                # BEFORE pointing -- look_at_image() uses the head's CURRENT
                # pose to compute the target angle (verified against the SDK
                # source, see robot_actions.py's module docstring). Do not
                # go_neutral() first, or this points at the wrong spot.
                goto_pose(mini, *chosen_pose)
                point_at(mini, outcome.chosen_pixel_center)

                width, _height = get_jpeg_size(chosen_frame)
                location_phrase = describe_location(outcome.chosen_pixel_center, width)
                say(mini, f"{outcome.chosen_label}, {location_phrase}")
            else:
                # Nothing matched after the full scan (always_act) or
                # after the clarifying answer (ask_when_ambiguous) --
                # say so instead of silently doing nothing. A trial
                # finishing in total silence reads as the system being
                # broken, not as a legitimate "not found" outcome.
                say(mini, "I don't see anything matching that request.")
                # No chosen object to pull a frame from -- fall back to
                # the center-view scan frame (always present) so this
                # trial still logs exactly one image, like every other.
                image_bytes_to_save = frames.get((0, 0)) if frames else None

            go_neutral(mini)  # reset for the next trial's scan

    except Exception as e:
        error = repr(e)

    end_time = datetime.datetime.now().isoformat(timespec="seconds")
    elapsed = round(time.time() - t0, 2)
    chosen_label = outcome.chosen_label if outcome else ""
    success = compute_success(chosen_label, correct_object)
    failure_reason = "other (exception)" if error and success != "1" else ""
    image_path = save_trial_image(image_bytes_to_save, participant_id, trial_id, start_time)

    row = {
        "participant_id": participant_id,
        "condition": condition,
        "trial_id": trial_id,
        "request_text": request_text,
        "label": label,
        "mode": mode,
        "candidates": "|".join(interpretation.get("candidates", []) or []),
        "is_ambiguous_model": interpretation.get("is_ambiguous"),
        "asked_question": outcome.asked_question if outcome else "",
        "question_text": outcome.question_text if outcome else "",
        "answer_text": outcome.answer_text if outcome else "",
        "chosen_label": chosen_label,
        "correct_object": correct_object or "",
        "success": success,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": elapsed,
        "console_wait_seconds": timing_sink.get("console_wait_seconds", 0),
        "extra_turns": outcome.extra_turns if outcome else 0,
        "failure_reason": failure_reason,
        "error": error,
        "image_path": image_path,
    }
    write_log_row(row)
    print(f"trial={trial_id} mode={mode} chosen={row['chosen_label']!r} "
          f"correct={correct_object!r} success={success} asked={row['asked_question']} "
          f"elapsed={elapsed}s image={image_path!r} error={error!r}")
    return row


def run_batch(csv_path, mode, participant_id, use_robot, image_path):
    mini = None
    cm = None
    jpeg_bytes = None

    if use_robot:
        from reachy_mini import ReachyMini
        from config import ROBOT_IP
        cm = ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend="default")
        mini = cm.__enter__()
        mini.media.start_playing()
        time.sleep(2.0)  # let webrtc / camera warm up
    elif image_path:
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
    else:
        raise ValueError("Batch mode needs either --robot or --image.")

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if use_robot:
                    print(f"\n=== Trial {row['trial_id']} ({row['label']}) ===")
                    print(f"Have the participant say: \"{row['request_text']}\"")
                    input("Press Enter once they've said it -- this starts the trial... ")

                # On the live robot, run_one_trial does its own scan
                # (locate_object) and ignores jpeg_bytes; it's only used
                # on the --image dev/debug path.
                run_one_trial(
                    jpeg_bytes,
                    row["request_text"],
                    row["label"],
                    mode,
                    row.get("correct_object"),
                    trial_id=row["trial_id"],
                    participant_id=participant_id,
                    condition=row.get("condition", ""),
                    mini=mini,
                )

                if use_robot:
                    input("Press Enter to continue to the next trial... ")
    finally:
        if mini is not None:
            mini.media.stop_playing()
            cm.__exit__(None, None, None)


# Per the experiment guide: C1 is Always Act + Clear, C2 is Always Act +
# Ambiguous, C3 is Ask When Ambiguous + Ambiguous. requests.csv carries
# `mode` per-row already (C1/C2 rows -> always_act, C3 rows -> ask_when_ambiguous),
# so run_session() doesn't need a --mode argument at all -- it just runs
# each condition's rows with whatever mode that row specifies.
def condition_order_for(participant_id: str) -> list:
    """C1 always runs first (the baseline condition). C2/C3 order then
    alternates by participant per the guide's counterbalancing table
    (P01/P03: C1->C2->C3, P02/P04: C1->C3->C2) -- generalized here by
    odd/even participant number so it still works for a 5th+ participant.
    Falls back to C1->C2->C3 if no number is found in participant_id.
    """
    m = re.search(r"\d+", participant_id or "")
    n = int(m.group()) if m else 1
    return ["C1", "C2", "C3"] if n % 2 == 1 else ["C1", "C3", "C2"]


def run_session(csv_path, participant_id, use_robot, image_path):
    """Runs one participant through the full C1/C2/C3 protocol in their
    counterbalanced order. requests.csv must have a `condition` column
    (C1/C2/C3) in addition to request_text/label/mode/correct_object. A
    `setup_hint` column, if present, is shown to the facilitator before
    each trial (which physical objects to place on the table).
    """
    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    order = condition_order_for(participant_id)
    print(f"\n##### Participant {participant_id!r}: condition order {' -> '.join(order)} #####")

    mini = None
    cm = None
    jpeg_bytes = None

    if use_robot:
        from reachy_mini import ReachyMini
        from config import ROBOT_IP
        cm = ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend="default")
        mini = cm.__enter__()
        mini.media.start_playing()
        time.sleep(2.0)
    elif image_path:
        with open(image_path, "rb") as f:
            jpeg_bytes = f.read()
    else:
        raise ValueError("Session needs either --robot or --image.")

    try:
        for condition in order:
            rows = [r for r in all_rows if r.get("condition") == condition]
            if not rows:
                print(f"(no rows found for condition {condition} in {csv_path} -- skipping)")
                continue

            print(f"\n======= Starting condition {condition} ({len(rows)} trials) =======")
            if use_robot:
                input(f"Press Enter when ready to begin {condition}... ")

            for row in rows:
                mode = row["mode"]
                if use_robot:
                    print(f"\n=== Trial {row['trial_id']} [{condition}] ({row['label']}) ===")
                    setup_hint = row.get("setup_hint")
                    if setup_hint:
                        print(f"SETUP: {setup_hint}")
                        input("Press Enter once the objects are in place... ")
                    print(f"Have the participant say: \"{row['request_text']}\"")
                    input("Press Enter once they've said it -- this starts the trial... ")

                # On the live robot, run_one_trial does its own scan
                # (locate_object) and ignores jpeg_bytes; it's only used
                # on the --image dev/debug path.
                run_one_trial(
                    jpeg_bytes,
                    row["request_text"],
                    row["label"],
                    mode,
                    row.get("correct_object"),
                    trial_id=row["trial_id"],
                    participant_id=participant_id,
                    condition=condition,
                    mini=mini,
                )

                if use_robot:
                    input("Press Enter to continue to the next trial... ")
    finally:
        if mini is not None:
            mini.media.stop_playing()
            cm.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to a JPEG for offline/dev testing (no robot)")
    parser.add_argument("--robot", action="store_true", help="Use the live Reachy Mini camera + motion")
    parser.add_argument("--request", help="The spoken request text (single-trial mode)")
    parser.add_argument("--mode", choices=["always_act", "ask_when_ambiguous"], default="always_act")
    parser.add_argument("--label", choices=["clear", "ambiguous"], default="clear")
    parser.add_argument("--correct-object", default=None, help="The predetermined ground-truth target, e.g. 'red cup'")
    parser.add_argument("--participant", default="", help="Participant ID, e.g. P1")
    parser.add_argument("--batch", help="Path to a requests CSV to run every row from, using --mode for all of them (dev/debug)")
    parser.add_argument("--session", help="Path to a requests CSV with a `condition` column -- runs the full C1/C2/C3 protocol for --participant")
    args = parser.parse_args()

    if args.session:
        if not args.participant:
            parser.error("--session needs --participant (e.g. P01) to determine the C2/C3 order.")
        run_session(args.session, args.participant, use_robot=args.robot, image_path=args.image)
        return

    if args.batch:
        run_batch(args.batch, args.mode, args.participant, use_robot=args.robot, image_path=args.image)
        return

    if args.robot:
        from reachy_mini import ReachyMini
        from config import ROBOT_IP
        with ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend="default") as mini:
            mini.media.start_playing()
            time.sleep(2.0)
            # jpeg_bytes=None is fine here -- run_one_trial does its own
            # scan (locate_object) whenever mini is set.
            run_one_trial(None, args.request, args.label, args.mode,
                          args.correct_object, trial_id="manual",
                          participant_id=args.participant, mini=mini)
            mini.media.stop_playing()
    else:
        if not args.image:
            parser.error("Need --image (offline) or --robot (live).")
        with open(args.image, "rb") as f:
            jpeg_bytes = f.read()
        run_one_trial(jpeg_bytes, args.request, args.label, args.mode,
                      args.correct_object, trial_id="manual",
                      participant_id=args.participant, mini=None)


if __name__ == "__main__":
    main()