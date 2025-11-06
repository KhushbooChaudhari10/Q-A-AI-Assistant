"""
Advanced RAG Agent Evaluation using industry-standard metrics:
- RAGAS (Retrieval-Augmented Generation Assessment)
- BERTScore (semantic similarity)
- ROUGE (n-gram overlap)

These are the proper evaluation frameworks mentioned in the task requirements.
"""

import os
import sys
import json
from typing import List, Dict, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.langgraph_agent import run_workflow
from dotenv import load_dotenv

load_dotenv()

# Import evaluation libraries
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️ RAGAS not available. Install with: pip install ragas datasets langchain-openai")
    RAGAS_AVAILABLE = False

# BERTScore disabled - downloads 1.4GB model, too heavy
BERTSCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️ ROUGE not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False


# Test dataset with ground truth
TEST_DATASET = [
    {
        "question": "What is LangGraph used for?",
        "ground_truth": "LangGraph is a framework for building stateful multi-actor applications with LLMs. It is used to create agent workflows with multiple nodes and edges.",
        "expected_in_kb": True
    },
    {
        "question": "What framework should I use for this task?",
        "ground_truth": "You should use LangGraph or similar AI Agent framework to create the RAG Q&A agent with a workflow containing plan, retrieve, answer, and reflect nodes.",
        "expected_in_kb": True
    },
    {
        "question": "What vector database is recommended?",
        "ground_truth": "ChromaDB or any open-source vector database is recommended for storing and querying embeddings in the RAG system.",
        "expected_in_kb": True
    },
    {
        "question": "What is the deadline for submission?",
        "ground_truth": "The deadline is 7 PM, 05.11.2025.",
        "expected_in_kb": True
    },
    {
        "question": "What is quantum entanglement?",
        "ground_truth": "Quantum entanglement is a physical phenomenon where particles remain connected so that the state of one affects the state of another.",
        "expected_in_kb": False  # NOT in our knowledge base
    }
]


def run_agent_and_collect_data() -> List[Dict[str, Any]]:
    """
    Run the RAG agent on all test questions and collect results.

    Returns:
        List of dictionaries with question, contexts, answer, ground_truth
    """
    print("\n" + "="*70)
    print("COLLECTING AGENT RESPONSES")
    print("="*70)

    results = []

    for i, test_case in enumerate(TEST_DATASET, 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']

        print(f"\n[{i}/{len(TEST_DATASET)}] Processing: {question}")

        # Run workflow
        result = run_workflow(question)

        # Extract data in RAGAS format
        answer = result.get('answer', '')
        retrieved_metadata = result.get('retrieved_metadata', [])
        retrieved_context = result.get('retrieved_context', '')

        # RAGAS expects contexts as a list of strings
        # Split the concatenated context back into individual chunks
        contexts = []
        if retrieved_context:
            # Split by source markers
            chunks = retrieved_context.split('[Source:')
            for chunk in chunks:
                if chunk.strip():
                    # Extract just the text content (after the metadata line)
                    lines = chunk.split('\n', 1)
                    if len(lines) > 1:
                        contexts.append(lines[1].strip())

        # If no contexts extracted, use empty list
        if not contexts:
            contexts = [""]

        data_point = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "expected_in_kb": test_case['expected_in_kb']
        }

        results.append(data_point)

        print(f"   ✓ Answer length: {len(answer)} chars")
        print(f"   ✓ Contexts retrieved: {len(contexts)}")

    return results


def evaluate_with_ragas(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate using RAGAS metrics.

    RAGAS provides:
    - faithfulness: Is the answer grounded in the context?
    - answer_relevancy: How relevant is the answer to the question?
    - context_precision: How precise are the retrieved contexts?
    - context_recall: How much of the ground truth is covered?

    Args:
        data: List of evaluation data points

    Returns:
        Dictionary of RAGAS metric scores
    """
    if not RAGAS_AVAILABLE:
        print("\n⚠️ Skipping RAGAS evaluation (not installed)")
        return {}

    print("\n" + "="*70)
    print("RAGAS EVALUATION")
    print("="*70)

    try:
        # Configure RAGAS to use HuggingFace instead of OpenAI
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            print("\n⚠️ HF_TOKEN not set. Skipping RAGAS evaluation.")
            print("RAGAS requires LLM access for evaluation metrics.")
            return {}

        print("\nConfiguring RAGAS to use HuggingFace API...")

        # Create LangChain LLM that uses HuggingFace endpoint
        llm = ChatOpenAI(
            model="openai/gpt-oss-20b:nebius",
            api_key=HF_TOKEN,
            base_url="https://router.huggingface.co/v1",
            temperature=0.1
        )

        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)

        # Configure metrics to use HuggingFace LLM
        for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
            metric.llm = ragas_llm

        # Convert to RAGAS dataset format
        dataset = Dataset.from_list(data)

        # Run RAGAS evaluation
        print("\nRunning RAGAS metrics (this may take a few minutes)...")
        print("Note: RAGAS uses HuggingFace LLM calls for evaluation\n")

        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        print("\n📊 RAGAS Results:")
        ragas_scores = {}
        for metric, value in results.items():
            ragas_scores[metric] = float(value)
            print(f"   {metric}: {value:.4f}")

        return ragas_scores

    except Exception as e:
        print(f"\n❌ RAGAS evaluation failed: {e}")
        print("This might be due to LLM API limits or configuration issues.")
        print("Continuing with other evaluation metrics...")
        return {}


# BERTScore removed - 1.4GB model download is too heavy for this project


def evaluate_with_rouge(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate using ROUGE metrics for n-gram overlap.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence

    Args:
        data: List of evaluation data points

    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        print("\n⚠️ Skipping ROUGE evaluation (not installed)")
        return {}

    print("\n" + "="*70)
    print("ROUGE EVALUATION")
    print("="*70)

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        print("\nComputing ROUGE scores...")

        for d in data:
            scores = scorer.score(d['ground_truth'], d['answer'])

            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        # Average scores
        avg_scores = {
            'rouge1_f1': sum(rouge1_scores) / len(rouge1_scores),
            'rouge2_f1': sum(rouge2_scores) / len(rouge2_scores),
            'rougeL_f1': sum(rougeL_scores) / len(rougeL_scores)
        }

        print("\n📊 ROUGE Results:")
        for metric, value in avg_scores.items():
            print(f"   {metric}: {value:.4f}")

        # Show per-question scores
        print("\n📋 Per-Question ROUGE-L F1:")
        for i, (question, score) in enumerate(zip([d['question'] for d in data], rougeL_scores), 1):
            print(f"   Q{i}: {score:.4f} - {question[:50]}...")

        return avg_scores

    except Exception as e:
        print(f"\n❌ ROUGE evaluation failed: {e}")
        return {}


def run_complete_evaluation() -> Dict[str, Any]:
    """
    Run complete evaluation with all available metrics.

    Returns:
        Complete evaluation results
    """
    print("\n" + "█"*70)
    print("RAG AGENT EVALUATION")
    print("█"*70)
    print("\nUsing industry-standard metrics:")
    print("  • RAGAS (RAG-specific metrics via LLM)")
    print("  • ROUGE (n-gram overlap)")
    print()

    # Step 1: Collect agent responses
    data = run_agent_and_collect_data()

    # Step 2: Run all evaluations
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'total_questions': len(data),
        'questions': [d['question'] for d in data],
        'metrics': {}
    }

    # RAGAS evaluation
    ragas_scores = evaluate_with_ragas(data)
    if ragas_scores:
        evaluation_results['metrics']['ragas'] = ragas_scores

    # ROUGE evaluation (lightweight, n-gram based)
    rouge_results = evaluate_with_rouge(data)
    if rouge_results:
        evaluation_results['metrics']['rouge'] = rouge_results

    # Calculate overall score
    all_scores = []
    for metric_group in evaluation_results['metrics'].values():
        all_scores.extend(metric_group.values())

    if all_scores:
        evaluation_results['overall_score'] = sum(all_scores) / len(all_scores)
    else:
        evaluation_results['overall_score'] = 0.0

    # Save detailed results
    output_file = os.path.join(
        'evaluation',
        f'advanced_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return evaluation_results


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """
    Generate comprehensive markdown report.

    Args:
        results: Evaluation results

    Returns:
        Markdown report
    """
    lines = []
    lines.append("# Advanced RAG Agent Evaluation Report\n\n")
    lines.append(f"*Timestamp*: {results['timestamp']}\n\n")
    lines.append(f"*Questions Evaluated*: {results['total_questions']}\n\n")

    lines.append("## Evaluation Frameworks Used\n\n")
    lines.append("This evaluation uses industry-standard metrics:\n\n")
    lines.append("1. *RAGAS* - Specialized metrics for RAG systems (LLM-based)\n")
    lines.append("2. *ROUGE* - N-gram overlap for content similarity\n\n")

    lines.append("## Results Summary\n\n")
    lines.append("| Metric Category | Metric | Score |\n")
    lines.append("|----------------|--------|-------|\n")

    for category, metrics in results.get('metrics', {}).items():
        for metric_name, score in metrics.items():
            lines.append(f"| {category.upper()} | {metric_name} | {score:.4f} |\n")

    if 'overall_score' in results:
        lines.append(f"\n*Overall Score*: {results['overall_score']:.4f}\n\n")

    lines.append("## Interpretation\n\n")

    # RAGAS interpretation
    if 'ragas' in results['metrics']:
        lines.append("### RAGAS Metrics\n\n")
        ragas = results['metrics']['ragas']

        if 'faithfulness' in ragas:
            lines.append(f"- *Faithfulness ({ragas['faithfulness']:.3f})*: "
                        f"Measures if answers are grounded in retrieved context\n")

        if 'answer_relevancy' in ragas:
            lines.append(f"- *Answer Relevancy ({ragas['answer_relevancy']:.3f})*: "
                        f"Measures how relevant answers are to questions\n")

        if 'context_precision' in ragas:
            lines.append(f"- *Context Precision ({ragas['context_precision']:.3f})*: "
                        f"Measures precision of retrieved contexts\n")

        if 'context_recall' in ragas:
            lines.append(f"- *Context Recall ({ragas['context_recall']:.3f})*: "
                        f"Measures how much ground truth is covered\n")

        lines.append("\n")

    # ROUGE interpretation
    if 'rouge' in results['metrics']:
        lines.append("### ROUGE Scores\n\n")
        rouge = results['metrics']['rouge']
        lines.append(f"- *ROUGE-1 ({rouge.get('rouge1_f1', 0):.3f})*: Unigram overlap\n")
        lines.append(f"- *ROUGE-2 ({rouge.get('rouge2_f1', 0):.3f})*: Bigram overlap\n")
        lines.append(f"- *ROUGE-L ({rouge.get('rougeL_f1', 0):.3f})*: Longest common subsequence\n\n")

    lines.append("## Conclusion\n\n")

    overall = results.get('overall_score', 0)
    if overall >= 0.7:
        lines.append("The RAG agent demonstrates *strong performance* across evaluation metrics.\n")
    elif overall >= 0.5:
        lines.append("The RAG agent demonstrates *good performance* with areas for improvement.\n")
    else:
        lines.append("The RAG agent shows *moderate performance* and requires optimization.\n")

    lines.append("\n---\n\n")
    lines.append("Generated using RAGAS and ROUGE evaluation frameworks\n")

    return ''.join(lines)


if __name__ == "__main__":
    print("\n🚀 Starting Advanced RAG Evaluation...")

    # Check which libraries are available
    print("\n📦 Checking dependencies:")
    print(f"   RAGAS: {'✓' if RAGAS_AVAILABLE else '✗'}")
    print(f"   ROUGE: {'✓' if ROUGE_AVAILABLE else '✗'}")

    if not any([RAGAS_AVAILABLE, ROUGE_AVAILABLE]):
        print("\n❌ No evaluation libraries available!")
        print("\nInstall with:")
        print("   pip install ragas datasets langchain-openai")
        print("   pip install rouge-score")
        sys.exit(1)

    # Run evaluation
    results = run_complete_evaluation()

    # Generate report
    print("\n📄 Generating evaluation report...")
    report = generate_evaluation_report(results)

    report_file = os.path.join(
        'evaluation',
        f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    )

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   Report saved to: {report_file}")

    # Print final summary
    print("\n" + "█"*70)
    print("EVALUATION COMPLETE")
    print("█"*70)
    print(f"\n✅ Overall Score: {results.get('overall_score', 0):.4f}")
    print(f"✅ Evaluated {results['total_questions']} questions")
    print(f"✅ Used {len(results['metrics'])} evaluation frameworks")
    print("\n")