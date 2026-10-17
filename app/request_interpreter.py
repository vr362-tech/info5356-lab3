"""
request_interpreter.py -- the "LLM interprets the user's spoken request"
module. Given the request text and the current object list (from
scene_vlm.describe_scene), decides which object(s) the request could mean,
whether that's ambiguous (2+ candidates) or clear (exactly 1), and -- if
ambiguous -- a short clarifying question.

Deliberately a single plain prompted call with no calibration or
fine-tuning: that's the thing this whole project is testing (RQ2 -- is a
plain prompted decision accurate enough to be useful), so don't be tempted
to make this fancier.

RANKING: "candidates" is an ORDERED list, best match first -- policy.py's
run_always_act (and run_ask_when_ambiguous's non-ambiguous branch) takes
candidates[0] as the chosen object, so the prompt explicitly tells the
model to rank the request's object TYPE (the noun -- "book", "cup", "bowl")
above a color/adjective match. Without this, a request like "the red book"
could end up matching a "red cloth" (right color, wrong type of object)
ahead of a "blue book" (right type, wrong color) just because of whatever
order the model happened to list them in. Color/adjective words are still
useful to pick between two objects of the SAME type (e.g. "red book" vs
"blue book" when both really are books) -- they just shouldn't outrank type
on their own.

This only helps when the scene actually contains an object of the matching
type. If scene_vlm never detects anything labeled "book" in the first
place (e.g. it mis-saw a book as "cloth" or "item"), there's nothing of
that type to rank to the top -- that's a vision-recognition limitation in
scene_vlm, not something this file's prompt can fix.
"""
import json

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

INTERPRET_PROMPT_TEMPLATE = """A robot is looking at a table with these objects:
{objects_json}

A person just said: "{request_text}"

Decide which object(s) on the table this request could reasonably refer to.

When ranking candidates, the request's object TYPE (the noun -- e.g.
"book", "cup", "bowl") matters more than a color or other adjective in the
request. An object whose label's type matches that noun is a better match
than an object that only shares a color/adjective with the request but is
a different type of object -- a type match should always rank above a
color-only match. Only use color/adjective words to choose between two or
more objects of the SAME matching type (e.g. "red book" vs "blue book"
when both are books).

Example: objects are "red cloth" and "blue book"; request is "look at the
red book" -- list "blue book" before "red cloth": it matches the noun
"book" even though its color doesn't match, while "red cloth" matches only
the color and is the wrong type of object.

List "candidates" with the single best match FIRST -- whatever program
calls you treats candidates[0] as your top choice, so put your actual best
answer there, not just any matching object.

Respond with ONLY a JSON object, no other text -- do not include any
explanation or reasoning before or after it, in this exact shape:
{{
  "candidates": [list of "label" strings from the objects above that could match, BEST MATCH FIRST],
  "is_ambiguous": true or false (true if 2+ candidates, false if exactly 1),
  "clarifying_question": "a short question to disambiguate, or null if not ambiguous"
}}
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _extract_json_object(text: str):
    """Fallback for when the model explains itself before/after the JSON
    despite being told not to -- the object is usually still in there,
    so pull out the first "{" through the last "}" and try that instead
    of giving up. This is what recovers cases like the model writing a
    paragraph of reasoning and then a real {"candidates": [...], ...}
    at the end -- without it, that real candidate list gets thrown away
    and the trial wrongly looks like nothing was found. Returns None if
    that substring still isn't valid JSON."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _extract_text(response) -> str:
    """Pull the actual text out of the response, skipping any non-text
    blocks (e.g. thinking blocks) that can appear before or after it."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(
        f"No text block in response (stop_reason={response.stop_reason!r}, "
        f"blocks={[getattr(b, 'type', type(b)) for b in response.content]!r})"
    )


def interpret_request(request_text: str, objects: list) -> dict:
    objects_json = json.dumps(objects)
    prompt = INTERPRET_PROMPT_TEMPLATE.format(
        objects_json=objects_json, request_text=request_text
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _strip_code_fence(_extract_text(response))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        recovered = _extract_json_object(text)
        if recovered is not None:
            print("[request_interpreter] model added extra text around the JSON -- recovered it anyway")
            return recovered
        print(f"[request_interpreter] non-JSON reply, treating as no match: {text!r}")
        return {"candidates": [], "is_ambiguous": False, "clarifying_question": None}


if __name__ == "__main__":
    # Quick standalone test, no robot/camera needed:
    #   python request_interpreter.py "look at the red book"
    # Try it against this exact scenario (the one that motivated the noun
    # > adjective ranking above) to sanity-check the ordering:
    #   objects include a "red cloth" and a "blue book" -- candidates[0]
    #   should come back as "blue book", not "red cloth".
    import sys
    test_objects = [
        {"label": "red cloth", "pixel_center": [340, 210]},
        {"label": "blue book", "pixel_center": [540, 215]},
        {"label": "notebook", "pixel_center": [430, 300]},
    ]
    print(json.dumps(interpret_request(sys.argv[1], test_objects), indent=2))