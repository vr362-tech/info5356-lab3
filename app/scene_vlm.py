"""
scene_vlm.py -- the robot's "what's in front of me" step. Turns a camera
frame into a structured list of objects on the table, using Claude's
vision input. Everything downstream (the LLM request interpreter, and the
ask-vs-act decision) depends on this list being reasonably accurate, so
keep the prompt simple and test it against a real photo of your actual
test table before trusting it on live requests.
"""
import base64
import json

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SCENE_PROMPT = """You are looking at a tabletop through a robot's camera.
List every distinct object you can see on the table. For each object, give:
- "label": a short name (e.g. "red cup", "blue cup", "book")
- "pixel_center": [u, v] -- your best estimate of the object's center point
  in the image, in pixels, counted from the top-left corner (u = across, v = down)

Respond with ONLY a JSON array, no other text -- do not include any
explanation or reasoning before or after it. Example:
[{"label": "red cup", "pixel_center": [340, 210]},
 {"label": "blue cup", "pixel_center": [540, 215]},
 {"label": "book", "pixel_center": [430, 300]}]
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _extract_json_array(text: str):
    """Fallback for when the model explains itself before/after the JSON
    despite being told not to -- the array is usually still in there, so
    pull out the first "[" through the last "]" and try that instead of
    giving up. Returns None if that substring still isn't valid JSON."""
    start = text.find("[")
    end = text.rfind("]")
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


def describe_scene(jpeg_bytes: bytes) -> list:
    """jpeg_bytes: raw JPEG bytes, e.g. from mini.media.get_frame_jpeg()
    or just `open(path, "rb").read()` for a test photo.

    Returns a list of {"label": str, "pixel_center": [u, v]} dicts, one
    per object detected on the table.
    """
    b64 = base64.standard_b64encode(jpeg_bytes).decode("utf-8")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": SCENE_PROMPT},
                ],
            }
        ],
    )
    text = _strip_code_fence(_extract_text(response))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        recovered = _extract_json_array(text)
        if recovered is not None:
            print("[scene_vlm] model added extra text around the JSON -- recovered it anyway")
            return recovered
        # Still nothing usable -- most often when the view genuinely has
        # nothing recognizable in it (e.g. an off-center scan angle
        # looking at empty table/wall). That's a normal outcome, not a
        # crash: treat it as "no objects seen" instead of blowing up the
        # whole scan. Printed so you can see what it actually said if
        # this keeps happening on views that DO have objects in them.
        print(f"[scene_vlm] non-JSON reply, treating as no objects seen: {text!r}")
        return []


if __name__ == "__main__":
    # Quick standalone test, no robot needed:
    #   python scene_vlm.py ../test_images/table.jpg
    import sys
    with open(sys.argv[1], "rb") as f:
        jpeg_bytes = f.read()
    print(json.dumps(describe_scene(jpeg_bytes), indent=2))