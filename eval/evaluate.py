"""
Accuracy/evaluation system for generated email replies.

Why not just one metric:
- Embedding cosine similarity to a gold reply rewards fluent paraphrase
  even when the reply is factually wrong (e.g. agrees to the wrong refund
  amount). It's a similarity signal, not a correctness signal.
- A single LLM-judge call is noisy (LLM judges are known to vary run-to-run)
  and gives no visibility into *why* it scored something a 3 vs a 4.

So this combines three independent signals:

1. BERTScore F1 (token-level contextual similarity, Zhang et al. 2020)
   -- more sensitive than sentence-embedding cosine to specific wrong details,
   since it matches at the token level rather than pooling everything into
   one vector.

2. Faithfulness (RAGAS-style, Es et al. 2023) -- decompose the generated
   reply into atomic claims, then have the LLM verify each claim is
   supported by the source email. Directly measures hallucination, which
   is the most concrete definition of "accurate" for this task.

3. G-Eval-style rubric judge (Liu et al. 2023) -- chain-of-thought scoring
   on relevance, correctness, tone, and completeness (1-5 each), run twice
   and averaged to reduce single-call variance (self-consistency /
   "jury" approach, Verga et al. 2024).

Final composite score is a weighted blend, documented in README.md.
"""
import json
import sys
from pathlib import Path
from statistics import mean

from bert_score import score as bert_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import complete_json

JUDGE_RUNS = 2  # self-consistency: average over N independent judge calls

FAITHFULNESS_SYSTEM = (
    "You are a strict fact-checker. You break a reply into atomic factual "
    "claims and verify each is supported by the source email. You are not "
    "evaluating quality or tone -- only whether claims are grounded."
)

FAITHFULNESS_PROMPT = """Source email:
{body}

Generated reply:
{reply}

Step 1: List the atomic factual claims made in the reply (specific commitments,
numbers, dates, statuses -- not generic pleasantries like "thank you for reaching out").
Step 2: For each claim, judge if it is "supported" (stated or reasonably inferable
from the source email) or "unsupported" (invented/hallucinated).

Return ONLY JSON:
{{
  "claims": [{{"claim": "...", "supported": true}}, ...],
  "faithfulness_score": <float 0-1, fraction of claims supported; 1.0 if no factual claims made>
}}"""

JUDGE_SYSTEM = (
    "You are an expert customer-support quality reviewer scoring AI-generated "
    "reply suggestions against a known-good reference reply. Think step by step "
    "before scoring -- your reasoning should justify each number."
)

JUDGE_PROMPT = """Customer email:
Subject: {subject}
Body: {body}

Reference (gold) reply, written by a human expert:
{gold_reply}

AI-generated candidate reply:
{generated_reply}

Score the candidate on each dimension from 1 (poor) to 5 (excellent), comparing
it against what the reference reply achieves. Give brief reasoning first, then scores.

Return ONLY JSON:
{{
  "reasoning": "...",
  "relevance": <1-5>,
  "correctness": <1-5>,
  "tone": <1-5>,
  "completeness": <1-5>
}}"""

WEIGHTS = {
    "bertscore": 0.25,
    "faithfulness": 0.35,
    "judge": 0.40,
}


def compute_bertscore(candidates: list[str], references: list[str]) -> list[float]:
    _, _, f1 = bert_score(candidates, references, lang="en", verbose=False)
    return [round(x, 4) for x in f1.tolist()]


def compute_faithfulness(body: str, reply: str) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(body=body, reply=reply)
    result = complete_json(prompt, system=FAITHFULNESS_SYSTEM, temperature=0.0)
    return result


def compute_judge_score(subject: str, body: str, gold_reply: str, generated_reply: str) -> dict:
    runs = []
    for _ in range(JUDGE_RUNS):
        prompt = JUDGE_PROMPT.format(
            subject=subject, body=body, gold_reply=gold_reply, generated_reply=generated_reply
        )
        runs.append(complete_json(prompt, system=JUDGE_SYSTEM, temperature=0.3))

    avg = {
        dim: round(mean(r[dim] for r in runs), 2)
        for dim in ["relevance", "correctness", "tone", "completeness"]
    }
    avg["judge_mean_1_5"] = round(mean(avg.values()), 2)
    avg["judge_normalized_0_1"] = round((avg["judge_mean_1_5"] - 1) / 4, 4)
    avg["runs"] = runs
    return avg


def evaluate_response(subject: str, body: str, gold_reply: str, generated_reply: str) -> dict:
    bertscore_f1 = compute_bertscore([generated_reply], [gold_reply])[0]
    faithfulness = compute_faithfulness(body, generated_reply)
    judge = compute_judge_score(subject, body, gold_reply, generated_reply)

    composite = (
        WEIGHTS["bertscore"] * bertscore_f1
        + WEIGHTS["faithfulness"] * faithfulness["faithfulness_score"]
        + WEIGHTS["judge"] * judge["judge_normalized_0_1"]
    )

    return {
        "bertscore_f1": bertscore_f1,
        "faithfulness_score": faithfulness["faithfulness_score"],
        "faithfulness_claims": faithfulness["claims"],
        "judge_scores": {k: v for k, v in judge.items() if k != "runs"},
        "judge_raw_runs": judge["runs"],
        "composite_score": round(composite, 4),
    }


def evaluate_all(results_path: Path, output_path: Path):
    records = json.loads(results_path.read_text())
    scored = []
    for r in records:
        print(f"  evaluating {r['id']}...")
        eval_result = evaluate_response(r["subject"], r["body"], r["gold_reply"], r["generated_reply"])
        scored.append({**r, "eval": eval_result})

    aggregate = {
        "n": len(scored),
        "avg_bertscore_f1": round(mean(s["eval"]["bertscore_f1"] for s in scored), 4),
        "avg_faithfulness": round(mean(s["eval"]["faithfulness_score"] for s in scored), 4),
        "avg_judge_normalized": round(mean(s["eval"]["judge_scores"]["judge_normalized_0_1"] for s in scored), 4),
        "avg_composite": round(mean(s["eval"]["composite_score"] for s in scored), 4),
        "weights": WEIGHTS,
    }

    output = {"aggregate": aggregate, "per_response": scored}
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nAggregate: {json.dumps(aggregate, indent=2)}")
    print(f"Wrote full results to {output_path}")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    evaluate_all(root / "results" / "generated_replies.json", root / "results" / "scores.json")
