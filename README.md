# INFO 5356 Lab 3 — Should the Robot Ask Before It Acts?

Part B starter: the LLM/VLM system on Reachy Mini that decides whether to
act immediately on an ambiguous spoken request, or ask a clarifying
question first.

## Pipeline

```
camera frame (JPEG)
   -> scene_vlm.describe_scene()         [Claude vision: what objects are on the table]
   -> request_interpreter.interpret_request()  [Claude text: candidates + ambiguous? + question]
   -> policy.run_always_act() / run_ask_when_ambiguous()
   -> robot_actions: go_neutral -> point_at(pixel) -> say(label)
   -> trial_runner logs everything to logs/trial_log.csv
```

## Setup

```
python3 -m venv venv

source venv/bin/activate   # for macOS/Linux user only

.\venv\Scripts\python.exe -m pip install -r requirements.txt  # for Windows user only
.\venv\Scripts\python.exe -m pip install reachy-mini

pip install -r requirements.txt
pip install reachy-mini    # already installed if you're reusing the Lab 2 venv

cp .env.example .env
# edit .env and paste your real ANTHROPIC_API_KEY (see note below)
```

Getting a key: go to console.anthropic.com (or platform.claude.com), sign
in, create an API key under your own account. This is separate from a
claude.ai chat subscription — it's pay-as-you-go by usage, and testing
today will cost cents. Don't commit `.env` — it's already in `.gitignore`.
If a teammate needs to run this locally too, just share the key value
with them directly (text/Slack), not through the repo.

## First thing to run (before touching the robot)

These two test the VLM + LLM calls in isolation and need nothing but a
photo and your API key:

```
cd app
python scene_vlm.py ../test_images/table.jpg
python request_interpreter.py "look at the cup"
```

Put an actual photo of your test table (red cup, blue cup, book, or
whatever your real objects are) at `test_images/table.jpg` first.

## Running a full trial

No robot (fastest iteration loop):
```
python trial_runner.py --image ../test_images/table.jpg \
    --request "look at the cup" --mode ask_when_ambiguous --label ambiguous
```

On the robot:
```
python trial_runner.py --robot --request "look at the cup" \
    --mode ask_when_ambiguous --label ambiguous
```

Batch (every row in requests.csv, same photo or same robot session,
compare both modes):
```
python trial_runner.py --batch ../requests.csv --mode always_act --image ../test_images/table.jpg
python trial_runner.py --batch ../requests.csv --mode ask_when_ambiguous --image ../test_images/table.jpg
```

`requests.csv` has 12 starter rows to get testing going today — the
final request list / trial count is part of Part D's experiment design,
so swap in their real list once it exists.

## Known gaps / what to do with remaining time

- The "ask" branch's answer-resolution (policy.py) is a word-overlap
  match, not a model call -- works for short answers like "the blue one"
  or "red", but upgrade it to a second small LLM call if you have time.
- Speech is macOS `say` + `afconvert` (see app/speech.py) -- zero setup,
  but Mac-only. Swap for `espeak` or an API TTS if someone's on
  Linux/Windows.
- `look_at_image(u, v)` turns the head toward a pixel coordinate but
  doesn't verify the robot ended up looking at the *right* object --
  that's a judgment call for whoever runs Part C's trials (does the head
  turn count as correct, or does it need to visibly settle on the object).
