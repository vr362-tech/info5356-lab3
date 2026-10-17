"""
policy.py -- the two experimental conditions (RQ1/RQ2):

  always_act          -- ignore ambiguity, pick the first candidate, act immediately.
  ask_when_ambiguous   -- if the interpreter flagged the request as ambiguous,
                          ask the clarifying question first, resolve to a
                          specific object once answered, then act.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class TrialOutcome:
    mode: str
    asked_question: bool
    question_text: Optional[str]
    answer_text: Optional[str]
    chosen_label: Optional[str]
    chosen_pixel_center: Optional[list]
    extra_turns: int = 0


def _find_object(objects: list, label: Optional[str]) -> Optional[dict]:
    """Looks up an object by label. Tries an exact match first, then falls
    back to word-overlap matching, since the scene VLM's label and the
    interpreter's chosen candidate don't always come back as the identical
    string (e.g. "iced coffee" vs "iced coffee cup with straw")."""
    if label is None or not objects:
        return None

    exact = next((o for o in objects if o["label"] == label), None)
    if exact:
        return exact

    label_words = set(label.lower().split())
    best_obj = None
    best_score = 0
    for o in objects:
        obj_words = set(o["label"].lower().split())
        score = len(obj_words & label_words)
        if score > best_score:
            best_score = score
            best_obj = o
    return best_obj


def run_always_act(interpretation: dict, objects: list) -> TrialOutcome:
    candidates = interpretation.get("candidates") or []
    chosen_label = candidates[0] if candidates else None
    chosen_obj = _find_object(objects, chosen_label)
    return TrialOutcome(
        mode="always_act",
        asked_question=False,
        question_text=None,
        answer_text=None,
        chosen_label=chosen_label,
        chosen_pixel_center=chosen_obj["pixel_center"] if chosen_obj else None,
    )


def run_ask_when_ambiguous(
    interpretation: dict, objects: list, get_answer: Callable[[str], str]
) -> TrialOutcome:
    candidates = interpretation.get("candidates") or []

    if not interpretation.get("is_ambiguous") or len(candidates) <= 1:
        chosen_label = candidates[0] if candidates else None
        chosen_obj = _find_object(objects, chosen_label)
        return TrialOutcome(
            mode="ask_when_ambiguous",
            asked_question=False,
            question_text=None,
            answer_text=None,
            chosen_label=chosen_label,
            chosen_pixel_center=chosen_obj["pixel_center"] if chosen_obj else None,
        )

    question = interpretation.get("clarifying_question") or "Which one did you mean?"
    answer_text = get_answer(question)

    answer_words = set(answer_text.lower().split())
    best_label = None
    best_score = 0
    for c in candidates:
        candidate_words = set(c.lower().split())
        score = len(candidate_words & answer_words)
        if score > best_score:
            best_score = score
            best_label = c
    chosen_label = best_label if best_score > 0 else candidates[0]

    chosen_obj = _find_object(objects, chosen_label)
    return TrialOutcome(
        mode="ask_when_ambiguous",
        asked_question=True,
        question_text=question,
        answer_text=answer_text,
        chosen_label=chosen_label,
        chosen_pixel_center=chosen_obj["pixel_center"] if chosen_obj else None,
        extra_turns=1,
    )