import numpy as np
from collections import defaultdict
from time import perf_counter
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModel
from itertools import product
import random

def multi_perspective_sampling(k, retrieved_texts, max_combinations=100):
    
    seed=1399
    
    print("Generating text embeddings with MedCPT...")
    
    model = AutoModel.from_pretrained("../../tmp/medcpt/MedCPT-Query-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("../../tmp/medcpt/MedCPT-Query-Encoder")
    
    texts = retrieved_texts

    with torch.no_grad():
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        )
        vectors = model(**encoded).last_hidden_state[:, 0, :].numpy()
    
    print(f"Generated {len(vectors)} embeddings with dimension {vectors.shape[1]}")

    print(f"Finding {k} clusters.")
    clusters = KMeans(n_clusters=k, random_state=seed).fit_predict(vectors)

    cluster_dict = defaultdict(list)
    for index, cluster in enumerate(clusters):
        cluster_dict[cluster].append(index)
    print("Clusters distribution:", dict(cluster_dict))

    cluster_sizes = [len(indices) for indices in cluster_dict.values()]
    total_combinations = np.prod(cluster_sizes)
    print(f"Total possible combinations: {total_combinations}")

    np.random.seed(seed)
    subsets = []
    
    if total_combinations <= max_combinations:
        print(f"Using all {total_combinations} combinations")
        all_combinations = list(product(*[indices for indices in cluster_dict.values()]))
        for combination in all_combinations:
            subsets.append([retrieved_texts[idx] for idx in combination])
    else:
        print(f"Sampling {max_combinations} random combinations out of {total_combinations}")
        for _ in range(max_combinations):
            subset = [
                np.random.choice(cluster_dict[cluster])
                for cluster in cluster_dict
            ]
            subsets.append([retrieved_texts[idx] for idx in subset])

    return subsets

