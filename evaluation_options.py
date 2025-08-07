import json
import re
import argparse
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from openai import OpenAI

class Args:
    def __init__(self, data_path, data_name, data_splits, batch_size):
        self.data_path = data_path
        self.data_name = data_name
        self.data_splits = data_splits
        self.batch_size = batch_size


def normalize_answer(answer, model_type):
    """Normalize the predicted answer based on the model type."""

    supported_models = ['flan', 'llama3', 'gpt3', 'gpt4', 'deepseek', 'medgemma']
    if model_type not in supported_models:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Expected one of: {supported_models}"
        )
    
    if model_type == 'flan':  
        return answer.split('-')[0].strip().replace('.', '').upper()
    
    elif model_type == 'llama3':  
        # Try to match formats like "Option B", "B", " B", "Option A\nExplanation..."
        match = re.search(r'\b(?:Option\s*)?([A-D])\b', answer, re.IGNORECASE)
        return match.group(1).upper() if match else answer.strip().upper()
    
    elif model_type == 'gpt3':
        match = re.search(r'[\*\n\s]*([A-D])[\.\s]', answer)
        return match.group(1) if match else answer
    
    elif model_type == 'gpt4':
        match = re.search(r'[\*\n\s]*([A-D])[\.\s]', answer)
        return match.group(1) if match else answer    
        # match = extract_final_choice(answer)
        # return match
    
    elif model_type == 'deepseek':
        match = re.search(r'The correct answer is[:\s\n]*\**\s*([A-D])[\.\s]', answer, re.IGNORECASE)
        return match.group(1).upper() if match else answer.strip().upper()
    
    elif model_type == 'medgemma':
        match = extract_final_choice(answer)
        return match

    return answer.strip().upper()


def compute_metrics(results, model_type):
    """Compute accuracy, precision, recall, and F1 score."""
    pred_answers = [record['llm_answer'] for record in results]
    
    correct_options = [record['correct_option'] for record in results]

    assert len(correct_options) == len(pred_answers)

    all_true = []
    all_pred = []

    for true_answer, pred_answer in zip(correct_options, pred_answers):

        normalized_pred = normalize_answer(pred_answer, model_type)

        valid_options = {'A', 'B', 'C', 'D'}
        if normalized_pred not in valid_options:
            normalized_pred = 'A'

        all_pred.append(normalized_pred)
        all_true.append(true_answer)

        print(pred_answer)
        print(normalized_pred)
        print(true_answer)
        print("="*25)
        print()

    cls_report = classification_report(all_true, all_pred, labels=['A', 'B', 'C', 'D'], target_names=['A', 'B', 'C', 'D'], output_dict=True)
    conf_matrix = confusion_matrix(all_true, all_pred, labels=['A', 'B', 'C', 'D'])
    print(cls_report)
    print(conf_matrix)

    return cls_report



def extract_final_choice(raw_output, modelname='gpt-4o-mini'):
    client = OpenAI(api_key="", base_url="") 

    prompt = f"""
You will be given a raw answer from a medical QA model. Your task is to extract only the final answer choice (A, B, C, or D) as it appears in the original text. 
Do not infer or guess. Do not output anything else. Just return the letter A, B, C, or D if present. If no such letter is mentioned, return the word None.

Raw output:
{raw_output}

Final answer:"""

    try:
        response = client.chat.completions.create(
            model=modelname,
            messages=[
                {"role": "system", "content": "You are an assistant in completing multiple choice questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1  # Ensure only A/B/C/D
        )
        gpt_final_answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"GPT extraction failed: {e}")

    return gpt_final_answer



def main(args):
    """Main function to load data, evaluate, and print results."""

    results = []
    with open(args.results_file, 'r', encoding='utf-8') as file:
        for line in file:
            results.append(json.loads(line))

    metrics_results = compute_metrics(results, args.model_type)
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
    parser.add_argument('--results_file', type=str, default='./results/MMLU_anatomy/pmc-llama-7b/vanilla/epoch_0_llm_answers.jsonl', help="Path to results JSONL file.")
    parser.add_argument('--model_type', type=str, default='pmc-llama', help="Model type (e.g., 'gpt', 'llama3', 'flan').")

    args = parser.parse_args()
    main(args)
