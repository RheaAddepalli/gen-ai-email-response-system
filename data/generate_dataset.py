"""
Generates a synthetic customer-support email dataset with paired
"gold" replies, across a fixed set of categories/tones so the
downstream retrieval + eval steps have real structure to work with.

Run: python data/generate_dataset.py
Output: data/emails.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import complete_json

CATEGORIES = [
    ("billing", "a customer disputing or asking about a charge"),
    ("technical_support", "a customer reporting a bug or feature not working"),
    ("refund_request", "a customer asking for a refund or cancellation"),
    ("sales_inquiry", "a prospective customer asking about pricing or features"),
    ("complaint", "a frustrated customer complaining about service quality"),
]
TONES = ["neutral", "frustrated", "urgent", "polite", "confused"]
N_PER_CATEGORY = 4  # 5 categories * 4 = 20 examples (kept small to stay well under API rate limits)

SYSTEM = (
    "You generate realistic, varied customer-support email datasets for "
    "testing AI reply-suggestion systems. Emails should read like real "
    "customers wrote them -- imperfect, specific, occasionally rambly."
)

PROMPT_TEMPLATE = """Generate ONE realistic customer support email and its ideal agent reply.

Category: {category} ({category_desc})
Customer tone: {tone}

The email should reference a SPECIFIC, concrete detail (an order number, a product name,
a dollar amount, a date, an error message -- invent something plausible) so downstream
evaluation of "faithfulness" has something to check against.

The gold_reply should be a professional, correct, complete response written by a
skilled support agent -- it must directly address the specific detail in the email,
not just generic reassurance.

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{
  "subject": "...",
  "body": "...",
  "gold_reply": "..."
}}"""


def generate_dataset():
    examples = []
    idx = 0
    for category, category_desc in CATEGORIES:
        for i in range(N_PER_CATEGORY):
            tone = TONES[i % len(TONES)]
            prompt = PROMPT_TEMPLATE.format(
                category=category, category_desc=category_desc, tone=tone
            )
            try:
                item = complete_json(prompt, system=SYSTEM, temperature=0.9)
            except Exception as e:
                print(f"  [skip] {category}/{tone}: {e}")
                continue
            item["id"] = f"{category}_{idx:03d}"
            item["category"] = category
            item["tone"] = tone
            examples.append(item)
            idx += 1
            print(f"  generated {item['id']}")
    return examples


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "emails.json"
    print(f"Generating {len(CATEGORIES) * N_PER_CATEGORY} synthetic emails...")
    data = generate_dataset()
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(data)} examples to {out_path}")
