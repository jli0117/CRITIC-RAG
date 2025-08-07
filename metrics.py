import re
import string
import itertools
import collections
import pdb
import numpy as np

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    #pdb.set_trace()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def extract_option(pred_answer):
    """Extracts the first valid option (A, B, C, or D) from the model's output."""
    try:
        pred_answer = pred_answer.strip() 
        match = re.search(r"^\s*([A-D])[\s\.\n]", pred_answer, re.IGNORECASE)
        return match.group(1).upper() if match else ""
    except Exception:  
        return ""

    


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def f1(samples, pred_answers):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers), "Samples and predictions length mismatch."

    total_precision = 0.0
    total_recall = 0.0
    num_valid = 0

    for sample, pred_answer in zip(samples, pred_answers):
        gold_option = sample['correct option']
        num_valid += 1
        
        #pred_tokens = set(normalize_answer(pred_answer))
        extracted_pred = extract_option(pred_answer)  # Extract first valid A/B/C/D
        if not extracted_pred:
            continue  # Skip if no valid prediction
        pred_tokens = set(normalize_answer(extracted_pred))
        print("predicted tokens: ", pred_tokens)

        gold_tokens = set(normalize_answer(gold_option))
        print("true tokens: ", gold_tokens)
        print('\n')

        common = pred_tokens.intersection(gold_tokens)

        precision = len(common) / (len(pred_tokens) + 1e-16)
        recall = len(common) / (len(gold_tokens) + 1e-16)

        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / (num_valid + 1e-16)
    avg_recall = total_recall / (num_valid + 1e-16)

    return avg_precision, avg_recall, (2 * avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-16)


def accuracy(samples, pred_answers):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0

    for sample, pred_answer in zip(samples, pred_answers):
        gold_options = sample['correct option']
        num_all_answers += 1

        # num_correct_answers += compute_exact(gold_options, pred_answer)
        extracted_pred = extract_option(pred_answer)  # Extract first valid A/B/C/D
        if extracted_pred:
            num_correct_answers += compute_exact(gold_options, extracted_pred)

        
    return num_correct_answers / (num_all_answers + 1e-16)
