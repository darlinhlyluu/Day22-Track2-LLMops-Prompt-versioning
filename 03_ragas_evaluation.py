"""Step 3: RAGAS evaluation for both prompt versions."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from config import DATA_DIR, EVIDENCE_DIR, configure_langsmith, ensure_dirs, require_api_settings

settings = configure_langsmith()

from ragas import evaluate

try:
    from ragas import EvaluationDataset, SingleTurnSample
except ImportError:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from config import make_chat_model, make_embeddings
from qa_pairs import QA_PAIRS
from rag_utils import build_vectorstore, get_rag_prompts, make_answer_chain, make_retriever

PROMPTS = get_rag_prompts()
METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def run_rag(retriever, answer_chain, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = answer_chain.invoke({"context": "\n\n".join(contexts), "question": question})
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list[dict]:
    retriever = make_retriever(vectorstore, k=3)
    answer_chain = make_answer_chain(PROMPTS[prompt_version], settings)
    results = []
    print(f"\nRunning {len(QA_PAIRS)} questions with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, answer_chain, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:70]}")
    return results


def build_ragas_dataset(rag_results: list[dict]):
    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["reference"],
        )
        for item in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _metric_values(result, name: str) -> list[float]:
    try:
        raw = result[name]
    except Exception:
        raw = result.to_pandas()[name].tolist()
    if isinstance(raw, (float, int)):
        raw = [raw]
    return [float(v) for v in raw if v is not None and not np.isnan(float(v))]


def run_ragas_eval(rag_results: list[dict], version: str) -> dict[str, float]:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=make_chat_model(settings, temperature=0.0),
        embeddings=make_embeddings(settings),
    )

    scores = {}
    for metric_name in METRIC_NAMES:
        values = _metric_values(result, metric_name)
        scores[metric_name] = float(np.mean(values)) if values else 0.0
        marker = " target" if metric_name == "faithfulness" and scores[metric_name] >= 0.8 else ""
        print(f"  {metric_name:20s}: {scores[metric_name]:.4f}{marker}")
    return scores


def comparison_lines(v1_scores: dict[str, float], v2_scores: dict[str, float]) -> list[str]:
    lines = []
    lines.append("=" * 72)
    lines.append("RAGAS comparison table")
    lines.append("=" * 72)
    lines.append(f"{'Metric':24s} {'V1':>10s} {'V2':>10s} {'Winner':>10s}")
    for metric in METRIC_NAMES:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2"
        lines.append(f"{metric:24s} {s1:10.4f} {s2:10.4f} {winner:>10s}")
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        lines.append(f"Target met: faithfulness = {best_faith:.4f}")
    else:
        lines.append(f"Below target: best faithfulness = {best_faith:.4f}")
    if v1_scores["faithfulness"] >= v2_scores["faithfulness"]:
        lines.append("Analysis: V1 is more concise, which often reduces unsupported claims.")
    else:
        lines.append("Analysis: V2's structured answer appears to improve grounding on this dataset.")
    return lines


def main() -> None:
    ensure_dirs()
    require_api_settings(settings)

    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore(settings)
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    lines = comparison_lines(v1_scores, v2_scores)
    print("\n".join(lines))

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "best_faithfulness": best_faith,
        "num_questions_per_prompt": len(QA_PAIRS),
        "analysis": lines[-1],
    }
    report_path = DATA_DIR / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (EVIDENCE_DIR / "03_ragas_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (EVIDENCE_DIR / "03_ragas_scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {report_path} and evidence/03_ragas_report.json")
    print("Capture the comparison table as evidence/03_ragas_scores.png.")


if __name__ == "__main__":
    main()

