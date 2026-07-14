# Hiver Challenge — Gen-AI Email Suggested-Response System

## Overview

This system takes an incoming customer support email and generates a suggested
reply using an LLM, grounded in a small dataset of past email→reply pairs via
hybrid retrieval. It then scores each generated reply with a multi-signal
accuracy system and reports both per-response and aggregate scores.

## 1. Dataset

**Source:** Synthetic, generated via LLM (`data/generate_dataset.py`) — 20
customer support emails across 5 categories (billing, technical support,
refund requests, sales inquiries, complaints), 4 examples each, spanning 5
tones (neutral, frustrated, urgent, polite, confused).

**Why synthetic:** No public dataset of real customer support emails paired
with high-quality "gold" agent replies is freely available at the size and
structure needed for this challenge without significant cleaning. Generating
synthetic data let me control structure (every email is seeded with a
concrete, checkable detail — an order number, dollar amount, or error code)
so that both the generator and the evaluator have something specific to be
right or wrong about, rather than generically polite text.

**Representativeness:** Each email follows a realistic pattern seen in real
support inboxes — a customer references a specific order/product/error, states
their issue, and (implicitly or explicitly) asks for resolution. Tone varies
so the generator has to handle more than one register. The gold replies are
written to directly resolve the stated issue, which is what makes the
faithfulness metric (below) meaningful — a reply that doesn't address the
specific detail is measurably worse, not just differently worded.

**Reused across runs:** The dataset was generated once and reused for all
downstream pipeline runs (`--skip-data`) to conserve free-tier API rate-limit
budget for the generation and evaluation stages, which is where this
challenge's core requirements live. `data/generate_dataset.py` is fully
functional and can regenerate a fresh dataset on demand.

## 2. Reply Generator

**Approach:** Hybrid retrieval + few-shot LLM generation (`generator/reply_generator.py`).

- **Retrieval:** fuses dense embeddings (`sentence-transformers`,
  `all-MiniLM-L6-v2`) with BM25 sparse retrieval, combined via **Reciprocal
  Rank Fusion**. Dense retrieval catches paraphrased/semantic matches; BM25
  catches exact terms (order numbers, product names) that embeddings tend to
  blur together. This is more robust than either method alone on short,
  keyword-heavy support text.
- **Generation:** the top-3 fused examples are used as few-shot context. The
  model is explicitly instructed to ground its reply only in details present
  in the *new* email, not to copy specifics from the examples — this
  constraint is what the faithfulness metric checks.

**Why not fine-tuning or a classifier:** Fine-tuning needs far more data than
a 20-example set can support, and a classical classifier can't generate
open-ended text. Prompting + retrieval was the right fit for the dataset size
and time constraints of this challenge, while still being "grounded" rather
than a naive zero-shot call.

**Trade-off:** retrieval quality depends on the dataset being representative
of incoming email types. At only 20 examples, retrieval has a small pool to
draw from — at production scale, this pool would be a much larger historical
reply corpus, separate from the evaluation set.

## 3. Accuracy / Evaluation System (core of this challenge)

**What "accurate" means here:** not exact match, and not just "sounds similar
to the gold reply." A fluent, well-toned reply that confidently states the
wrong refund amount is *not* accurate. I define accuracy as three things
combined:
1. Does it say something semantically aligned with what a good agent would say?
2. Is every claim it makes actually grounded in the source email (no invented
   promises, amounts, or facts)?
3. Would a careful reviewer judge it as relevant, correct, appropriately
   toned, and complete?

No single metric captures all three, so the system combines three
independent signals:

| Signal | What it catches | Weight |
|---|---|---|
| **BERTScore F1** | Token-level semantic overlap with the gold reply | 0.25 |
| **Faithfulness** (RAGAS-style claim decomposition) | Hallucinated commitments/numbers not supported by the source email | 0.35 |
| **LLM-jury rubric** (G-Eval-style chain-of-thought scoring, 2 runs averaged) | Relevance, correctness, tone, completeness vs. the gold reply | 0.40 |

**Why not just cosine similarity to the gold reply?** It rewards paraphrase,
not correctness — a reply can sound right and still commit to the wrong
number. Faithfulness (decomposing the reply into atomic claims and verifying
each against the source email) is weighted highest, along with the reasoning
judge, because those two most directly answer "is this actually accurate."
BERTScore is included because it's cheap, deterministic, and catches
token-level mismatches (wrong dates/numbers) that a coarser
sentence-embedding similarity would smooth over.

**Why the judge runs twice and is averaged:** single LLM-judge calls are
known to vary run-to-run. Averaging two independent scoring passes (a small
"jury") is a cheap variance-reduction step, following the self-consistency
idea used in G-Eval and jury-of-judges evaluation setups.

### Validating the metric against real judgment

I manually reviewed several scored responses to check the composite score's
ranking matched my own reading of quality, rather than trusting the number
blindly. A consistent pattern emerged: the generator's replies are
well-toned and polite (tone scores 4.5–5/5 across the board) but
systematically **less decisive** than the gold replies — they tend to ask a
clarifying question ("could you confirm...") instead of committing to a
resolution the way the human-written gold reply does. This was correctly
penalized: correctness and completeness scores clustered around 2–3/5, and
faithfulness scores were pulled down because claims like "we'll follow up" or
"we're investigating" aren't grounded in anything the generator actually
verified. This gave me confidence the scoring system is measuring something
real, not just rewarding fluent text — it's catching a genuine, consistent
weakness in the generator's behavior (hedging instead of resolving) that a
pure similarity metric would have missed entirely.

### Model note (rate-limit trade-off)

Development started with `llama-3.3-70b-versatile` (Groq), which has a tight
free-tier daily budget (100K tokens/day). After exhausting that quota
mid-testing, the pipeline was switched to `llama-3.1-8b-instant`, which has a
much larger free-tier allowance (500K tokens/day, 14,400 requests/day) and a
separate quota bucket. This was a deliberate engineering trade-off — smaller
model, but a workable budget for a live coding challenge. Observed effect:
switching models measurably shifted the scores (faithfulness dropped from
~0.63 to ~0.50, BERTScore rose slightly to ~0.92) — the smaller model
produces safer, more template-like language that surface-matches the gold
reply more closely, while being less reliable at making precise, grounded
claims. This is itself a useful illustration of why a single similarity
metric is insufficient: BERTScore alone would have suggested the weaker model
was doing *better*.

### Not implemented (considered trade-offs)

- **Self-critique/revision pass** (generate → LLM critiques its own draft →
  regenerate) was considered but not implemented, to conserve API rate-limit
  budget for the core generation + evaluation pipeline within the time limit.
  The existing judge prompt could be repurposed as a pre-submission critique
  step as a natural next iteration.
- Judge and faithfulness checks currently use the same model used for
  generation; a production version would use a separate, stronger model as
  judge to reduce self-preference bias.
- Retrieval pool is the dataset itself; at production scale this would be a
  separate, larger historical reply corpus.

## How I used AI tools

I used Claude (Anthropic) throughout development — to design the evaluation
architecture (choosing and combining BERTScore, RAGAS-style faithfulness
decomposition, and G-Eval-style judge scoring), to write and iterate on the
pipeline code, and to debug issues live during the build (a Groq daily
rate-limit exhaustion, and JSON parsing failures from a smaller model). The
architectural decisions — which signals to combine, how to weight them, which
model to use given rate-limit constraints, and what to cut given the time
limit — were mine; Claude was used as a coding and design-sounding-board tool,
not as an autonomous decision-maker.

## Results (this run)

20 examples, `llama-3.1-8b-instant`:

```json
{
  "avg_bertscore_f1": 0.9155,
  "avg_faithfulness": 0.4996,
  "avg_judge_normalized": 0.5654,
  "avg_composite": 0.6299
}
```

Full per-response breakdown (including per-claim faithfulness verification
and judge reasoning) is in `results/scores.json`.
## How to run

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd hiver-challenge

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
# Create a .env file in the repo root with:
# GROQ_API_KEY=gsk_your_key_here

# 5. Run the pipeline
python run_pipeline.py                # full run: generate dataset -> generate replies -> evaluate
python run_pipeline.py --skip-data    # reuse existing data/emails.json
python run_pipeline.py --limit 5      # quick smoke test on 5 examples
```

**Note:** first run downloads `roberta-large` (~1.4GB, used by BERTScore) and
`all-MiniLM-L6-v2` (small, used for retrieval embeddings) from Hugging Face.
Subsequent runs use the local cache and are much faster.

```bash
python run_pipeline.py                # full run: generate dataset -> generate replies -> evaluate
python run_pipeline.py --skip-data    # reuse existing data/emails.json
python run_pipeline.py --limit 5      # quick smoke test on 5 examples
```

**Note:** first run downloads `roberta-large` (~1.4GB, used by BERTScore) and
`all-MiniLM-L6-v2` (small, used for retrieval embeddings) from Hugging Face.
Subsequent runs use the local cache and are much faster.

Outputs:
- `data/emails.json` — the dataset
- `results/generated_replies.json` — generated replies + retrieval context
- `results/scores.json` — per-response scores and aggregate averages

## References
- Zhang et al., 2020 — BERTScore: Evaluating Text Generation with BERT
- Liu et al., 2023 — G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment
- Es et al., 2023 — RAGAS: Automated Evaluation of Retrieval Augmented Generation
- Verga et al., 2024 — Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models