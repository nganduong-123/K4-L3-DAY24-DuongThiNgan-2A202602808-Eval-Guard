from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, hashlib, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARTIFACTS_DIR, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, TEST_SET_PATH

EVALUATOR_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    zeros = {name: 0.0 for name in metric_names}
    zeros["per_question"] = []
    zeros["metadata"] = {"evaluator_models": []}

    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal length")
    if not questions:
        return zeros

    zero_per_question = [
        EvalResult(
            question=question,
            answer=answer,
            contexts=context,
            ground_truth=ground_truth,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )
        for question, answer, context, ground_truth in zip(
            questions, answers, contexts, ground_truths
        )
    ]
    zero_results = {**zeros, "per_question": zero_per_question}
    if not OPENAI_API_KEY:
        raise RuntimeError("RAGAS requires OPENAI_API_KEY for a live evaluation")

    def safe_score(value) -> float:
        try:
            score = float(value)
            if math.isnan(score):
                return 0.0
            # Some cosine-based RAGAS metrics can exceed 1 by machine epsilon
            # (for example 1.0000000000000004); keep the public score contract.
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.0

    try:
        # On Windows, importing pyarrow (through datasets) before PyTorch can make
        # c10.dll fail to initialize because both load native runtime libraries.
        # Load PyTorch first to keep the evaluator deterministic across platforms.
        import torch  # noqa: F401
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Groq's OpenAI-compatible endpoint supports a single completion per
        # request. One generated question is sufficient for this 50-row lab and
        # avoids RAGAS' default `n=3`, which that endpoint rejects.
        answer_relevancy.strictness = 1

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        run_config = RunConfig(
            timeout=int(os.getenv("RAGAS_TIMEOUT_SECONDS", "600")),
            max_retries=3,
            max_wait=60,
            max_workers=max(1, int(os.getenv("RAGAS_MAX_WORKERS", "4"))),
        )

        model_pool = [
            model.strip()
            for model in os.getenv("RAGAS_MODEL_POOL", OPENAI_MODEL).split(",")
            if model.strip()
        ]
        if not model_pool:
            raise ValueError("RAGAS_MODEL_POOL must contain at least one model")
        while len(model_pool) < len(metrics):
            model_pool.append(model_pool[-1])

        # A model pool allows each metric to use a separate provider quota. The
        # per-model limiter prevents free-tier 429s without hiding failed rows.
        rpm = float(os.getenv("RAGAS_RPM_PER_MODEL", "0"))
        rate_limiters = {}
        for metric, model_name in zip(metrics, model_pool):
            llm_kwargs = {
                "api_key": OPENAI_API_KEY,
                "model": model_name,
                "temperature": 0,
                "max_tokens": int(os.getenv("RAGAS_MAX_TOKENS", "384")),
                "timeout": int(os.getenv("RAGAS_REQUEST_TIMEOUT_SECONDS", "90")),
                "max_retries": 2,
            }
            if OPENAI_BASE_URL:
                llm_kwargs["base_url"] = OPENAI_BASE_URL
            if rpm > 0:
                llm_kwargs["rate_limiter"] = rate_limiters.setdefault(
                    model_name,
                    InMemoryRateLimiter(
                        requests_per_second=rpm / 60.0,
                        check_every_n_seconds=0.1,
                        max_bucket_size=1,
                    ),
                )
            metric.llm = LangchainLLMWrapper(
                ChatOpenAI(**llm_kwargs), run_config=run_config
            )

        embeddings = HuggingFaceEmbeddings(
            model_name=EVALUATOR_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        batch_size = max(1, int(os.getenv("RAGAS_BATCH_SIZE", "10")))
        checkpoint_path = os.path.join(ARTIFACTS_DIR, "ragas_checkpoint.json")
        input_payload = {
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
        }
        input_hash = hashlib.sha256(json.dumps(
            input_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")).hexdigest()
        score_records = []
        checkpoint_models = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, encoding="utf-8") as file:
                checkpoint = json.load(file)
            previous_pool = [
                model.strip()
                for model in os.getenv("RAGAS_CHECKPOINT_MODEL_POOL", "").split(",")
                if model.strip()
            ]
            compatible_hashes = {input_hash}
            # Backward compatibility for checkpoints made before the hash was
            # separated from the evaluator pool. A previous pool must be
            # explicitly supplied, so scores can never be attached to a
            # different dataset by accident.
            for pool in (model_pool[:len(metrics)], previous_pool):
                if pool:
                    legacy_payload = {**input_payload, "models": pool}
                    compatible_hashes.add(hashlib.sha256(json.dumps(
                        legacy_payload, ensure_ascii=False, sort_keys=True
                    ).encode("utf-8")).hexdigest())
            if checkpoint.get("input_hash") in compatible_hashes:
                saved_scores = checkpoint.get("scores", [])
                if isinstance(saved_scores, list) and len(saved_scores) <= len(dataset):
                    score_records = saved_scores
                    checkpoint_models = checkpoint.get("models", []) or previous_pool

        batch_attempts = max(1, int(os.getenv("RAGAS_BATCH_ATTEMPTS", "4")))
        raise_metric_exceptions = os.getenv(
            "RAGAS_RAISE_EXCEPTIONS", "true"
        ).strip().lower() not in {"0", "false", "no"}
        for start in range(len(score_records), len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch = dataset.select(range(start, stop))
            last_error = None
            for attempt in range(1, batch_attempts + 1):
                try:
                    result = evaluate(
                        batch,
                        metrics=metrics,
                        embeddings=embeddings,
                        run_config=run_config,
                        raise_exceptions=raise_metric_exceptions,
                    )
                    break
                except Exception as error:
                    last_error = error
                    if attempt == batch_attempts:
                        raise
                    # A whole batch must succeed before it is checkpointed. This
                    # avoids silently turning transient API failures into zeros.
                    time.sleep(min(60, 5 * (2 ** (attempt - 1))))
            else:  # pragma: no cover - the final attempt raises above
                raise RuntimeError(f"RAGAS batch failed: {last_error}")
            frame = result.to_pandas()
            for _, row in frame.iterrows():
                score_records.append({
                    name: safe_score(row.get(name)) for name in metric_names
                })
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            temporary_path = checkpoint_path + ".tmp"
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "input_hash": input_hash,
                        "models": list(dict.fromkeys(
                            checkpoint_models + model_pool[:len(metrics)]
                        )),
                        "scores": score_records,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(temporary_path, checkpoint_path)

        per_question = [
            EvalResult(
                question=str(questions[index]),
                answer=str(answers[index]),
                contexts=list(contexts[index]),
                ground_truth=str(ground_truths[index]),
                faithfulness=safe_score(scores.get("faithfulness")),
                answer_relevancy=safe_score(scores.get("answer_relevancy")),
                context_precision=safe_score(scores.get("context_precision")),
                context_recall=safe_score(scores.get("context_recall")),
            )
            for index, scores in enumerate(score_records)
        ]
        aggregates = {
            name: (
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in metric_names
        }
        return {
            **aggregates,
            "per_question": per_question,
            "metadata": {
                "evaluator_models": list(dict.fromkeys(
                    checkpoint_models
                    + model_pool[:len(metrics)]
                    + [EVALUATOR_EMBEDDING_MODEL]
                ))
            },
        }
    except Exception as error:
        raise RuntimeError(f"Live RAGAS evaluation failed: {error}") from error


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời có chi tiết không được hỗ trợ bởi ngữ cảnh.",
            "Siết system prompt, chỉ cho phép dùng context và đặt temperature=0.",
        ),
        "context_recall": (
            "Retriever bỏ sót đoạn chứa đủ thông tin cần thiết.",
            "Điều chỉnh chunking hoặc tăng tín hiệu từ khóa của BM25.",
        ),
        "context_precision": (
            "Các đoạn ít liên quan đang đứng quá cao trong kết quả tìm kiếm.",
            "Cải thiện cross-encoder reranking hoặc lọc theo metadata.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa đi thẳng vào trọng tâm câu hỏi.",
            "Viết lại prompt để yêu cầu trả lời trực tiếp và ngắn gọn.",
        ),
    }
    analyzed = []
    for item in eval_results:
        scores = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "contexts": item.contexts,
            "ground_truth": item.ground_truth,
            "average_score": sum(scores.values()) / len(scores),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analyzed.sort(key=lambda item: item["average_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
