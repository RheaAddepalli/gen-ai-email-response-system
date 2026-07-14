"""
Thin wrapper around the Groq API so the rest of the codebase doesn't
care which model/provider is behind it. Swap MODEL below if needed.
"""
import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def complete(prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Single-turn completion with retry, including rate-limit-aware backoff."""
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if attempt == max_attempts - 1:
                raise
            if is_rate_limit:
                wait = getattr(e, "retry_after", None) or (5 * (attempt + 1))
                print(f"  [rate limited] waiting {wait}s before retry (attempt {attempt + 1}/{max_attempts})...")
                time.sleep(wait)
            else:
                time.sleep(1.5 * (attempt + 1))

def complete_json(prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.3) -> dict:
    """Ask for strict JSON back and parse it, retrying if the model returns malformed JSON
    (smaller/faster models occasionally break on unescaped quotes inside string values)."""
    json_instruction = (
        "\n\nCRITICAL: Return ONLY valid JSON. No markdown fences, no preamble. "
        "Any double quote character that appears INSIDE a string value must be escaped as \\\". "
        "Double-check your output is valid JSON before finishing."
    )
    last_error = None
    last_raw = None
    for attempt in range(3):
        raw = complete(
            prompt + json_instruction,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature if attempt == 0 else 0.0, 
        )
        last_raw = raw
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  [json parse failed, attempt {attempt + 1}/3] {e}")
    raise ValueError(f"complete_json: failed to parse valid JSON after 3 attempts. Last error: {last_error}\nLast raw output: {last_raw}")