import os
import json
import torch
import faiss
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import biorag.query_encoder as qe
import biorag.retrieve as rt
import biorag.rerank as rr
import pdb

def run(samples):

    embeddings_dir = '../../tmp/rag/embeddings'
    articles_dir = '../../tmp/rag/articles'
        
    # params for efficiently loading the large pubmed corpus
    pubmed_group_num = 10
    max_load_num = 38

    # query preprocess
    input_list = qe.query_preprocess(samples)
    
    # query encode
    xq = qe.query_encode(input_list)

    # ========== Retrieve from PubMed ==========
    # pubmed mips
    pubmed_I_array = []
    for start_index in range(0, max_load_num, pubmed_group_num):
        pubmed_index = rt.pubmed_index_create(pubmed_embeddings_dir=os.path.join(embeddings_dir, "pubmed"), start_index=start_index, pubmed_group_num=pubmed_group_num)
        pubmed_I_array_temp = []
        splits = [i for i in range(0, len(xq), 1024)]

        for split_start in tqdm(splits, desc=f"PubMed FAISS MIPS {start_index}:"):
            D, I = pubmed_index.search(xq[split_start:split_start+1024], max_load_num)   
            pubmed_I_array_temp.extend(I)
        pubmed_I_array.append(pubmed_I_array_temp)
        del pubmed_index
    print(len(pubmed_I_array), "x", len(pubmed_I_array[0]))
    
    # # pubmed mips index save
    # np.save("PubMed_I_array.npy", pubmed_I_array)

    # pubmed decode
    pubmed_evidences = rt.pubmed_decode(pubmed_I_array, pubmed_articles_dir= os.path.join(articles_dir, "pubmed"), pubmed_group_num=pubmed_group_num)
    print(len(pubmed_evidences), "x", len(pubmed_evidences[0]))

    # ========== Retrieve from CPG ==========

    # cpg mips
    cpg_index = rt.cpg_index_create(cpg_embeddings_dir = os.path.join(embeddings_dir, "cpg"))
    cpg_I_array = []

    splits = [i for i in range(0, len(xq), 1024)]

    for i in tqdm(splits, desc="CPG FAISS MIPS"):
        D, I = cpg_index.search(xq[i:i+1024], 10)   
        cpg_I_array.extend(I)
    del cpg_index

    # decode cpg
    cpg_evidences = rt.cpg_decode(cpg_I_array, cpg_articles_dir = os.path.join(articles_dir, "cpg"))


    # ========== Retrieve from Textbook ==========
    # textbook mips
    textbook_index = rt.textbook_index_create(textbook_embeddings_dir = os.path.join(embeddings_dir, "textbook"))
    textbook_I_array = []

    splits = [i for i in range(0, len(xq), 1024)]

    for i in tqdm(splits, desc="textbook FAISS MIPS"):
        D, I = textbook_index.search(xq[i:i+1024], 10)   
        textbook_I_array.extend(I)
    del textbook_index

    # decode textbook
    textbook_evidences = rt.textbook_decode(textbook_I_array, textbook_articles_dir = os.path.join(articles_dir, "textbook"))

    # ========== Combine and Rerank ==========
    query_evidences, evidences, sources = rr.combine_query_evidence(input_list, pubmed_evidences, cpg_evidences, textbook_evidences)

    reranked_10evidences, reranked_10indices, reranked_10scores, reranked_10sources = rr.rerank(
        query_evidences, evidences, sources
    )

    return reranked_10evidences, reranked_10indices, reranked_10sources, reranked_10scores
