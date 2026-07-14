"""
End-to-end pipeline: dataset -> generate replies -> evaluate -> report.

Usage:
    export GROQ_API_KEY=sk-...
    python run_pipeline.py                 # full run: generate data if missing, then generate+eval
    python run_pipeline.py --skip-data      # reuse existing data/emails.json
    python run_pipeline.py --limit 10       # quick smoke test on N examples
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def ensure_dataset(skip_data: bool):
    data_path = ROOT / "data" / "emails.json"
    if skip_data and data_path.exists():
        print(f"[1/3] Reusing existing dataset ({data_path})")
        return json.loads(data_path.read_text())
    print("[1/3] Generating synthetic dataset...")
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_dataset.py")], check=True)
    return json.loads(data_path.read_text())


def generate_all_replies(corpus, limit: int = None):
    from generator.reply_generator import HybridRetriever, generate_reply

    print("[2/3] Building retriever and generating replies...")
    retriever = HybridRetriever(corpus)
    subset = corpus[:limit] if limit else corpus

    results = []
    for i, item in enumerate(subset):
        out = generate_reply(retriever, item["subject"], item["body"], exclude_id=item["id"])
        results.append({**item, **out})
        print(f"  [{i+1}/{len(subset)}] generated reply for {item['id']}")

    out_path = ROOT / "results" / "generated_replies.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Wrote {len(results)} generated replies to {out_path}")
    return out_path


def run_eval(generated_path: Path):
    from eval.evaluate import evaluate_all

    print("[3/3] Running evaluation (BERTScore + faithfulness + LLM-jury)...")
    scores_path = ROOT / "results" / "scores.json"
    return evaluate_all(generated_path, scores_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N examples (smoke test)")
    args = parser.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)

    corpus = ensure_dataset(args.skip_data)
    generated_path = generate_all_replies(corpus, limit=args.limit)
    output = run_eval(generated_path)

    print("\n=== DONE ===")
    print(f"Aggregate composite score: {output['aggregate']['avg_composite']}")
    print("Full breakdown in results/scores.json")


if __name__ == "__main__":
    main()
