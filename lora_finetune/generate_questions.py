"""
Generates a larger, diverse synthetic patient-question set via Claude to scale
up the LoRA training data beyond the small curated + scraped question pool.

Output matches the schema TestCaseGenerator.load_forum_cases() already reads
(test_data/*.json with a "questions" list), so prepare_dataset.py picks this
file up automatically - nothing else needs to change.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

QUESTIONS_PER_CATEGORY = 12
OUTPUT_PATH = Path("test_data/synthetic_test_cases.json")

CATEGORY_DESCRIPTIONS = {
    "symptom_identification": "patient describing symptoms and asking what they might mean",
    "treatment_options": "asking about home treatment or management options",
    "lifestyle": "diet, exercise, daily habits related to prevention/management",
    "when_to_see_doctor": "uncertain whether symptoms warrant a doctor visit (mix of true red flags and benign concerns)",
    "diagnosis_request": "asking the bot to diagnose their specific condition",
    "medication_request": "asking about specific medications or prescriptions",
    "surgery_question": "asking about surgical options or necessity",
    "anxiety": "worried/scared, possibly fearing cancer or serious illness",
    "embarrassment": "embarrassed to discuss with a doctor or others",
    "treatment_efficacy": "follow-up on whether a treatment is working",
    "worsening": "symptoms getting worse despite treatment",
    "multiple_concerns": "combining multiple questions or conditions at once",
    "misinformation": "asking to confirm/debunk a myth they heard",
    "prevention": "asking how to prevent issues, possibly in a specific population (pregnancy, postpartum, elderly, etc.)",
}

GENERATION_PROMPT = """Generate {n} realistic, diverse patient questions about hemorrhoids and/or constipation for the category "{category}".

Category meaning: {description}

Guidelines:
- Write as an actual patient would ask (casual language, first person, sometimes anxious or embarrassed)
- Vary length, tone, and specificity across the {n} questions
- Do not repeat phrasing patterns from question to question
- Return ONLY a JSON array of {n} strings, no other text, no markdown fences"""


def generate_category_questions(client, category: str, n: int) -> list:
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                n=n, category=category,
                description=CATEGORY_DESCRIPTIONS[category],
            ),
        }],
    )
    text = next(block.text for block in message.content if block.type == "text")
    json_str = text[text.find("["):text.rfind("]") + 1]
    return json.loads(json_str)


def main():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    questions = []
    for category in CATEGORY_DESCRIPTIONS:
        print(f"Generating {QUESTIONS_PER_CATEGORY} questions for '{category}'...")
        try:
            texts = generate_category_questions(client, category, QUESTIONS_PER_CATEGORY)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        for i, text in enumerate(texts, 1):
            questions.append({
                "id": f"synthetic_{category}_{i:03d}",
                "title": text.strip(),
                "body": "",
                "source": "synthetic",
                "url": "",
                "category": category,
                "added_date": datetime.now().isoformat(),
            })
        print(f"  +{len(texts)} questions")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_date": datetime.now().isoformat(),
            "total_questions": len(questions),
            "collection_method": "synthetic_llm_generated",
            "questions": questions,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(questions)} synthetic questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
