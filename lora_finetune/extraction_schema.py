"""
Shared schema for the red-flag/triage JSON-extraction fine-tuning task.

Mirrors the flag taxonomy and urgency tiers already used by
patient_chatbot.py's RED_FLAG_PATTERNS / create_red_flag_warning, so a
model fine-tuned on this schema is a drop-in upgrade path for that
regex-based logic, not just a benchmark exercise.
"""
import json
import re
from typing import Dict, List

RED_FLAGS = [
    "severe_pain",
    "heavy_bleeding",
    "fever",
    "prolonged_constipation",
    "black_stool",
    "dizziness",
]

# Same grouping as patient_chatbot.create_red_flag_warning's critical/non_urgent sets.
CRITICAL_FLAGS = {"heavy_bleeding", "black_stool", "severe_pain", "dizziness"}
NON_URGENT_FLAGS = {"prolonged_constipation", "fever"}

URGENCY_LEVELS = ["emergency", "see_doctor_soon", "routine"]

FLAG_DESCRIPTIONS = {
    "severe_pain": "severe/excruciating/unbearable pain, not routine discomfort",
    "heavy_bleeding": 'heavy or significant rectal bleeding, blood clots, "filling the toilet"',
    "fever": "fever, chills, or feeling hot and cold",
    "prolonged_constipation": "no bowel movement for 3 or more days",
    "black_stool": "black, tarry, or coffee-ground-colored stool",
    "dizziness": "dizziness, faintness, lightheadedness, or passing out",
}

EXTRACTION_SYSTEM_PROMPT = f"""You are a medical triage classifier for a hemorrhoids/constipation patient chatbot.

Given a patient's message, identify which of these red-flag symptoms are present, then output ONLY compact JSON (no markdown, no other text) matching this schema:
{{"red_flags": [<subset of {RED_FLAGS}>], "urgency": "<emergency|see_doctor_soon|routine>", "reasoning": "<one short sentence>"}}

Red flag definitions:
{chr(10).join(f"- {flag}: {desc}" for flag, desc in FLAG_DESCRIPTIONS.items())}

Urgency rules:
- "emergency" if ANY of heavy_bleeding, black_stool, severe_pain, dizziness is present
- "see_doctor_soon" if only prolonged_constipation and/or fever are present (no critical flags)
- "routine" if no red flags are present

If no red flags apply, return {{"red_flags": [], "urgency": "routine", "reasoning": "..."}}."""


def urgency_from_flags(flags: List[str]) -> str:
    flagset = set(flags)
    if flagset & CRITICAL_FLAGS:
        return "emergency"
    if flagset & NON_URGENT_FLAGS:
        return "see_doctor_soon"
    return "routine"


def extract_json(text: str) -> Dict:
    """Best-effort parse of a schema-shaped JSON object out of a model response.

    Urgency is always recomputed from the extracted flags rather than trusted
    verbatim from the model, since it's a deterministic function of them -
    this removes one axis of label noise at both labeling and eval time.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")
    candidate = re.sub(r"[\x00-\x1f]+", " ", text[start:end + 1])
    parsed = json.loads(candidate)
    flags = [f for f in parsed.get("red_flags", []) if f in RED_FLAGS]
    return {
        "red_flags": sorted(set(flags)),
        "urgency": urgency_from_flags(flags),
        "reasoning": str(parsed.get("reasoning", ""))[:300],
    }
