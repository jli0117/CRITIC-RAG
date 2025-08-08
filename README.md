# CRITIC-RAG: Verification-Enhanced Retrieval-Augmented Generation for Medical QA

CRITIC-RAG is a novel framework designed to enhance the factual accuracy and reliability of large language models (LLMs) in the domain of medical question answering. By integrating an instruction-tuned verifier throughout the RAG pipeline, CRITIC-RAG systematically filters irrelevant information, verifies answer grounding, and enables structured multi-step reasoning.

## 🧠 Key Features

- **Retrieve-on-Demand**: Determines whether external retrieval is necessary based on query complexity.
- **Evidence Filtering**: Filters retrieved content using a verifier model to ensure relevance.
- **Self-Consistency via Clustering**: Clusters evidence and samples diverse subsets for multi-path reasoning.
- **Groundedness Verification**: Selects the most reliable answer based on evidence alignment scores.
- **Plug-and-Play**: Designed for easy integration with a variety of LLMs without major architectural changes.


## 🧪 Benchmarks Used

- **[MedQA](https://arxiv.org/abs/2309.06024)**  
  A large-scale dataset with over 12,000 multiple-choice questions from USMLE-style exams covering various medical subfields.

  > [Github](https://github.com/jind11/MedQA)


- **[MMLU-Med (Medical Subset of MMLU)](https://arxiv.org/abs/2009.03300)**  
  Part of the Massive Multitask Language Understanding benchmark. Includes:
  - anatomy  
  - clinical knowledge  
  - college biology  
  - college medicine  
  - medical genetics  
  - professional medicine  
  > [HuggingFace](https://huggingface.co/datasets/cais/mmlu)

- **[LiveQA (TREC 2017 Medical QA)](https://trec.nist.gov/data/medical.html)**  
  Real-world consumer health questions from the TREC LiveQA medical track.
  > [GitHub](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017)

- **[MedicationQA](https://aclanthology.org/2023.acl-long.610/)**  
  Open-domain dataset of 4,151 questions focused on medication usage, dosage, interactions, and side effects.
  > [Github](https://github.com/abachaa/Medication_QA_MedInfo2019)

---

## 🏗️ Architecture

Below is a visual overview of the CRITIC-RAG pipeline, highlighting its multi-stage verification components:

![CRITIC-RAG Pipeline](critic-rag-pipeline.png)


## 🔗 Code References

This project builds on ideas and code from the following repositories:

- [KALMV](https://github.com/JinheonBaek/KALMV): Knowledge-Augmented Language Model Verification (EMNLP 2023)  
- [Self-BioRAG](https://github.com/dmis-lab/self-biorag): Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models (Bioinformatics, 2024)  


