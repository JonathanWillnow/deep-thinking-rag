"""
Evaluation metrics for the Deep Thinking RAG pipeline.

This module implements evaluation using the RAGAs library to
measure retrieval and generation quality.
"""

from typing import List, Dict, Any, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
import pandas as pd


def create_evaluation_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str]
) -> Dataset:
    """
    Create a RAGAs-compatible evaluation dataset.

    Args:
        questions: List of user queries.
        answers: List of generated answers.
        contexts: List of context lists (one per question).
        ground_truths: List of ground truth answers.

    Returns:
        A HuggingFace Dataset object for evaluation.
    """
    eval_data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    return Dataset.from_dict(eval_data)


def run_ragas_evaluation(
    dataset: Dataset,
    metrics: Optional[List] = None
) -> Dict[str, Any]:
    """
    Run RAGAs evaluation on the provided dataset.

    Args:
        dataset: The evaluation dataset.
        metrics: Optional list of metrics to use. Defaults to standard set.

    Returns:
        Dictionary containing evaluation results.
    """
    if metrics is None:
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_correctness,
        ]

    print("Running RAGAs evaluation...")
    result = evaluate(dataset, metrics=metrics, is_async=False)
    print("Evaluation complete.")

    return result


def compare_pipelines(
    query: str,
    baseline_answer: str,
    baseline_contexts: List[str],
    advanced_answer: str,
    advanced_contexts: List[str],
    ground_truth: str
) -> pd.DataFrame:
    """
    Compare baseline and advanced RAG pipelines on the same query.

    Args:
        query: The evaluation query.
        baseline_answer: Answer from baseline RAG.
        baseline_contexts: Contexts used by baseline RAG.
        advanced_answer: Answer from advanced RAG.
        advanced_contexts: Contexts used by advanced RAG.
        ground_truth: The ground truth answer.

    Returns:
        DataFrame with comparison results.
    """
    dataset = create_evaluation_dataset(
        questions=[query, query],
        answers=[baseline_answer, advanced_answer],
        contexts=[baseline_contexts, advanced_contexts],
        ground_truths=[ground_truth, ground_truth]
    )

    result = run_ragas_evaluation(dataset)

    results_df = result.to_pandas()
    results_df.index = ['baseline_rag', 'deep_thinking_rag']

    print("\n--- RAGAs Evaluation Results ---")
    print(results_df[['context_precision', 'context_recall', 'faithfulness', 'answer_correctness']].T)

    return results_df


def extract_contexts_from_state(state: Dict[str, Any]) -> List[str]:
    """
    Extract all retrieved contexts from a RAG state.

    Args:
        state: The final RAG state dictionary.

    Returns:
        List of context strings from all retrieval steps.
    """
    contexts = []

    past_steps = state.get('past_steps', [])
    for step in past_steps:
        synthesized = step.get('synthesized_context', '')
        if synthesized:
            contexts.append(synthesized)

    return list(set(contexts))  # Remove duplicates
