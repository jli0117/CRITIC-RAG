import json
import re
import argparse
import os
import numpy as np
from datetime import datetime

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer


class Args:
    def __init__(self, data_path, data_name, data_splits, batch_size):
        self.data_path = data_path
        self.data_name = data_name
        self.data_splits = data_splits
        self.batch_size = batch_size


def compute_metrics(results):
    """Compute accuracy, BLEU, ROUGE, and BERTScore for long texts."""
    
    pred_answers = [record['llm_answer'] for record in results]
    correct_answers = [record['correct_answer'] for record in results]
    
    assert len(pred_answers) == len(correct_answers)

    # ─────────────── BLEU ────────────────
    smoothie = SmoothingFunction().method4
    bleu_score = corpus_bleu(
        list_of_references=[[ref.split()] for ref in correct_answers],
        hypotheses=[pred.split() for pred in pred_answers],
        smoothing_function=smoothie
    )

    # ─────────────── ROUGE ───────────────
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    for ref, pred in zip(correct_answers, pred_answers):
        scores = rouge.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # ─────────────── BERTScore ───────────────
    scorer = BERTScorer(
        model_type="../../models/biobert-v1.1",  # Local path to BioBERT
        lang="en",
        rescale_with_baseline=False,
        num_layers=12
    )
    
    # This step assumes answers are short and in similar format.
    P, R, F1 = scorer.score(pred_answers, correct_answers)

    # Final result dictionary
    return {
        "bleu": bleu_score,
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        "bert_score_precision": P.mean().item(),
        "bert_score_recall": R.mean().item(),
        "bert_score_f1": F1.mean().item(),
    }



def main(args):
    """Main function to load data, evaluate, and print results."""

    results = []
    with open(args.results_file, 'r', encoding='utf-8') as file:
        for line in file:
            results.append(json.loads(line))

    metrics_results = compute_metrics(results)
    print(metrics_results)

    # Save results in logs.txt
    results_dir = os.path.dirname(args.results_file)
    log_file_path = os.path.join(results_dir, "logs.txt")


    # Prepare log entry with timestamp
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_results
    }

    # Append to log file
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "="*40 + "\n")  # Separator for readability
        json.dump(log_entry, log_file, indent=4)
        log_file.write("\n" + "="*40 + "\n")  # Closing separator

    print(f"Metrics successfully logged in: {log_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate QA model performance.")
    parser.add_argument('--data_path', type=str, default='./datasets', help="Path to the dataset.")
    parser.add_argument('--data_name', type=str, default='MMLU_anatomy', help="Dataset name.")
    parser.add_argument('--data_splits', type=str, default='train,test', help="Comma-separated data splits.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loading.")
    parser.add_argument('--results_file', type=str, default='./', help="Path to results JSONL file.")

    args = parser.parse_args()
    main(args)
