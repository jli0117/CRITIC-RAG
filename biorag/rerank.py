import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def combine_query_evidence(queries, list1, list2, list3):
    evidences_4 = []
    sources_4 = []

    for sublist1, sublist2, sublist3 in zip(list1, list2, list3):
        combined_evidence = sublist1 + sublist2 + sublist3
        combined_sources = (
            ['pubmed'] * len(sublist1) +
            ['cpg'] * len(sublist2) +
            ['textbook'] * len(sublist3)
        )
        evidences_4.append(combined_evidence)
        sources_4.append(combined_sources)

    q_4a_list = []
    for ith, q in tqdm(enumerate(queries), desc="Pairing query with evidence"):
        q_4a = []
        for a in evidences_4[ith]:
            q_4a.append([q, a])  # pair (query, evidence)
        q_4a_list.append(q_4a)

    return q_4a_list, evidences_4, sources_4



def rerank(q_4a_list, evidences_4, sources_4):
    device_ids = [0] if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained("../../tmp/medcpt/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("../../tmp/medcpt/MedCPT-Cross-Encoder")
    model = model.to(device_ids[0])

    logits_list = []
    for q_4a in tqdm(q_4a_list, desc="Reranking..."):
        with torch.no_grad():
            encoded_q_4a = tokenizer(
                q_4a,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded_q_4a = {key: tensor.to(device_ids[0]) for key, tensor in encoded_q_4a.items()}
            logits_q_4a = model(**encoded_q_4a).logits.squeeze(dim=1)
            logits_q_4a = logits_q_4a.detach().cpu()
            logits_list.append(logits_q_4a)

    sorted_indices = [
        sorted(range(len(logits_4)), key=lambda k: logits_4[k], reverse=True)
        for logits_4 in logits_list
    ]
    top_10_indices = [sorted_i[:10] for sorted_i in sorted_indices]

    sorted_evidence_list = []
    top_10_scores = []
    top_10_sources = []

    for index, data in enumerate(evidences_4):
        sorted_evidence_list.append([data[i] for i in top_10_indices[index]])
        top_10_scores.append([logits_list[index][i] for i in top_10_indices[index]])
        top_10_sources.append([sources_4[index][i] for i in top_10_indices[index]])

    return sorted_evidence_list, top_10_indices, top_10_scores, top_10_sources
