import os
import json
import pdb
from typing import Dict
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader

FOLDER_NAMES = {
    'MMLU_anatomy': 'mmlu_anatomy',
    'MMLU_clinical_knowledge': 'mmlu_clinical_knowledge',
    'MMLU_college_biology': 'mmlu_college_biology',
    'MMLU_college_medicine': 'mmlu_college_medicine',
    'MMLU_medical_genetics': 'mmlu_medical_genetics',
    'MMLU_professional_medicine': 'mmlu_professional_medicine',
    'MedQA':'medqa',
    'MedMCQA': 'medmcqa_demo',
    'UltraMedical': 'ultramedical',
    'MMLU_merged': 'mmlu_merged',
    'LiveQA': 'liveqa',
    'MedicationQA': 'medicationqa'
}

class QADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind: int):
        return self.data[ind]

    def __iter__(self):
        for sample in self.data:
            yield sample

    @staticmethod
    def collate(x):
        return x

def get_qa_datasets(args, cpu_usage_ratio=1.0):
    datasets = {
        split: QADataset([
            json.loads(sample) for sample in \
            list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', f'{split}-samples.jsonl'), 'r'))
        ])
        for split in args.data_splits
    }
    data_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size = args.batch_size,
            shuffle = split == 'train' and not isinstance(dataset, torch.utils.data.IterableDataset),
            num_workers = int(cpu_count() * cpu_usage_ratio),
            collate_fn = datasets[split].collate
        )
        for split, dataset in datasets.items()
    }
    return datasets, data_loaders

