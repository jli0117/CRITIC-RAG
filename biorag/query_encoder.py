import torch
import sys
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import json
import pdb

def query_preprocess(input_data): 

    def combine_q_o(sample):
        prompt = f"Question: {sample['question']}"
        if 'options' in sample:
            prompt += " Options: "
            for key, value in sample['options'].items():
                prompt += f"{key}. {value} "
        return prompt
    
    query_list = []
    for sample in input_data:
        query = combine_q_o(sample)
        query_list.append(query)    

    return query_list


def query_encode(input_list):
    model = AutoModel.from_pretrained("../../tmp/medcpt/MedCPT-Query-Encoder")
    if torch.cuda.is_available():
        model = model.to(0)
    tokenizer = AutoTokenizer.from_pretrained("../../tmp/medcpt/MedCPT-Query-Encoder")

    queries=[]

    splits = [i for i in range(0, len(input_list), 100)]
    for i in tqdm(splits, desc="query encoding"):
        split_queries = input_list[i:i+100]
        with torch.no_grad():
            encoded = tokenizer(
                split_queries, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=192,
        )
            encoded = {key: tensor.to(0) for key, tensor in encoded.items()}
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            query_embeddings = embeds.detach().cpu().numpy()      
            queries.extend(query_embeddings)
            xq = np.vstack(queries)
    return xq