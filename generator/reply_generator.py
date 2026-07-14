"""
Suggested-reply generator.

Retrieval: hybrid dense (sentence-transformers embeddings) + sparse
(BM25) retrieval over a pool of past email->reply pairs, fused with
Reciprocal Rank Fusion (RRF). Hybrid retrieval consistently beats
pure-dense or pure-sparse on short, keyword-heavy text like support
emails (dense catches semantic/paraphrase matches, sparse catches
exact terms like order numbers or product names that embeddings blur).

Generation: the fused top-k examples are used as few-shot context for
the LLM, which is asked to write a reply grounded ONLY in the new
email's specific details (this constraint matters a lot for the
faithfulness metric in eval/evaluate.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import complete

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # fast, good enough for retrieval
TOP_K = 3
RRF_K = 60  # standard RRF damping constant


class HybridRetriever:
    def __init__(self, corpus: list[dict]):
        """corpus: list of {id, subject, body, gold_reply, category}"""
        self.corpus = corpus
        self.texts = [f"{c['subject']} {c['body']}" for c in corpus]

        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)

        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query_subject: str, query_body: str, k: int = TOP_K, exclude_id: str = None):
        query = f"{query_subject} {query_body}"

        # dense ranking
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ q_emb
        dense_rank = np.argsort(-dense_scores)

        # sparse ranking
        sparse_scores = self.bm25.get_scores(query.lower().split())
        sparse_rank = np.argsort(-sparse_scores)

        # reciprocal rank fusion
        rrf_scores = np.zeros(len(self.corpus))
        for rank_pos, idx in enumerate(dense_rank):
            rrf_scores[idx] += 1.0 / (RRF_K + rank_pos + 1)
        for rank_pos, idx in enumerate(sparse_rank):
            rrf_scores[idx] += 1.0 / (RRF_K + rank_pos + 1)

        order = np.argsort(-rrf_scores)
        results = []
        for idx in order:
            item = self.corpus[idx]
            if exclude_id and item["id"] == exclude_id:
                continue
            results.append(item)
            if len(results) == k:
                break
        return results


GENERATION_SYSTEM = (
    "You are an expert customer-support agent writing suggested email replies. "
    "You are shown similar past emails and their ideal replies as style/structure "
    "examples. Ground your reply ONLY in the specific details present in the NEW "
    "email -- do not invent order numbers, amounts, or promises that aren't stated "
    "or reasonably inferable. If a detail is missing, ask for it rather than guessing."
)

GENERATION_PROMPT = """Similar past examples (for tone/structure reference only -- do not copy their specific details):

{few_shot_block}

---
NEW customer email to reply to:
Subject: {subject}
Body: {body}

Write the suggested reply. Reply with ONLY the email body text, no subject line, no preamble."""


def format_few_shot(examples: list[dict]) -> str:
    blocks = []
    for ex in examples:
        blocks.append(
            f"Email: {ex['subject']} -- {ex['body']}\nIdeal reply: {ex['gold_reply']}"
        )
    return "\n\n".join(blocks)


def generate_reply(retriever: HybridRetriever, subject: str, body: str, exclude_id: str = None) -> dict:
    examples = retriever.retrieve(subject, body, k=TOP_K, exclude_id=exclude_id)
    few_shot_block = format_few_shot(examples)
    prompt = GENERATION_PROMPT.format(few_shot_block=few_shot_block, subject=subject, body=body)
    reply = complete(prompt, system=GENERATION_SYSTEM, temperature=0.5)
    return {
        "generated_reply": reply,
        "retrieved_context_ids": [ex["id"] for ex in examples],
    }


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "emails.json"
    corpus = json.loads(data_path.read_text())
    retriever = HybridRetriever(corpus)

    # smoke test on the first email, excluding itself from retrieval
    sample = corpus[0]
    out = generate_reply(retriever, sample["subject"], sample["body"], exclude_id=sample["id"])
    print("SUBJECT:", sample["subject"])
    print("GENERATED REPLY:\n", out["generated_reply"])
    print("RETRIEVED CONTEXT:", out["retrieved_context_ids"])
