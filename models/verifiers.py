import numpy as np
import os
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

from metrics import normalize_answer, accuracy, f1

import pdb

METRIC_NAMES = {
    'accuracy': accuracy,
    'f1': f1
}


INSTRUCTIONS_isRET = [
    """You are given a multiple-choice question related to a question-answering task.

    Question: {question}

    Options:
    A. This is a question related to medical domain, which may require external knowledge sources.
    B. This is a common sense question, which do not require external knowledge sources.

    Please select one option:""",

    """You are tasked with determining whether an external database retrieval is necessary to answer the following question.

    Question: {question}

    Options:
    A. External database retrieval is required to answer this question.
    B. External database retrieval is not required to answer this question.

    Select one option:""",
    """
    Below is a question that may or may not require external database retrieval for a complete answer.

    Question: {question}

    Options:
    A. External database retrieval is required to answer this question.
    B. External database retrieval is not required to answer this question.

    Please select one option:""",
    """You are presented with a multiple-choice question and need to assess the necessity of external database retrieval for answering it.

    Question: {question}

    Options:
    A. External database retrieval is required to answer this question.
    B. External database retrieval is not required to answer this question.

    Choose one option:""",
    
    """Your task is to evaluate whether external database retrieval is essential for providing an answer to the following question.

    Question: {question}

    Options:
    A. External database retrieval is required to answer this question.
    B. External database retrieval is not required to answer this question.

    Please select one option:""",

    """Analyze the following question and determine whether background information from an external source is necessary to provide an accurate answer.

    Question: {question}

    Options:
    A. This question requires retrieving information from an external database to be answered correctly.
    B. This question can be answered without accessing any external database.

    Select one option:"""
   
]


INSTRUCTIONS_isREL = [
    """You are presented with a question-answering task. A passage has been retrieved from external databases, and you need to assess whether it helps answer the given question.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Please select the most appropriate option:""",
    
    """Consider the following question and passage. The passage has been retrieved from external sources, and your task is to evaluate its relevance and usefulness for answering the question.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Choose the correct option:""",
    
    """You are given a question and a passage retrieved from an external database. Your goal is to determine if the passage can be useful in answering the question.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Select the correct option:""",
    
    """Below is a question and a passage retrieved from external sources. Assess the relevance and helpfulness of the passage in answering the question.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Please choose the best option:""",
    
    """In this task, you are provided with a question and a passage. The passage has been retrieved from external sources, and you need to determine if it helps in answering the question.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Select the option that best describes the passage:""",

    """Evaluate the following question and passage. The passage was retrieved to assist in answering the question. Your task is to judge whether it is both relevant and useful.

    Question: {question}
    Passage: {passage}

    Options:
    A. The passage is irrelevant to answering the question, or relevant but not helpful for answering the question.
    B. The passage is both relevant and helpful for answering the question.

    Indicate your choice by selecting the most accurate option:"""
]


INSTRUCTIONS_isGRD = [
    """Please rate for the following question-answering task. An AI model has generated a response based on a specific question, and a relevant passage has been retrieved from external databases to aid in formulating the answer. Your task is to assess the accuracy of the generated output, particularly focusing on whether it is adequately supported by the retrieved content.
    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options: 
    A: Poor (not supported at all); 
    B: Fair (some support, but significant issues); 
    C: Good (moderate support, some flaws); 
    D: Very Good (strong support, minor flaws); 
    E: Excellent (fully supported and accurate)

    Select one option:""",
    
    """Please rate for the following question-answering task. An AI model has generated a response based on a specific question, and a relevant passage has been retrieved from external databases to aid in formulating the answer. Your task is to assess the accuracy of the generated output, particularly focusing on whether it is adequately supported by the retrieved content. 
    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options: 
    A: Completely unsupported (the answer contradicts or lacks evidence from the passage).
    B: Weakly supported (some alignment with the passage, but major factual gaps or inconsistencies remain).
    C: Partially supported (generally consistent with the passage but contains moderate inaccuracies or omissions).
    D: Mostly supported (strong alignment with the passage, though minor flaws or missing details exist).
    E: Fully supported (the answer is entirely consistent with and well-grounded in the retrieved passage).

    Select one option:""", 

    """You are given a question, a passage retrieved from external sources, and an answer generated by an AI model. Your task is to evaluate how well the answer is supported by the content of the passage.

    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options:
    A: Not supported at all; 
    B: Weakly supported with major flaws; 
    C: Somewhat supported but contains notable issues; 
    D: Largely supported with minor issues; 
    E: Fully supported and accurate.

    Choose the most appropriate rating:""",

    """Assess the following question-answering instance. The AI-generated answer was produced using the given question and a retrieved passage. Rate how accurately the passage supports the answer.

    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options:
    A: Unsupported — no meaningful connection between answer and passage;
    B: Limited support — some weak relevance but major factual problems;
    C: Moderate support — generally relevant with some inaccuracies;
    D: Strong support — minor flaws only;
    E: Complete support — accurate and fully grounded in the passage.

    Select one option:""",

    """Review the question, the passage retrieved from an external source, and the AI’s answer. Rate the degree to which the passage supports the generated answer.

    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options:
    A: The answer is not supported by the passage;
    B: The answer has minimal support with clear inconsistencies;
    C: The answer is somewhat supported but has moderate issues;
    D: The answer is mostly supported with only minor flaws;
    E: The answer is fully supported and accurate.

    Select the most fitting rating:""",

    """In this task, you are asked to score the answer generated by an AI system. The answer was based on a question and a supporting passage retrieved from external sources. Focus your rating on how well the passage supports the answer.

    Question: {question}
    Passage: {passage}
    Output: {answer}

    Options:
    A: No support — the answer is irrelevant or incorrect;
    B: Weak support — loosely connected, but many issues;
    C: Partial support — some accurate elements but also notable flaws;
    D: Good support — mostly correct with minor issues;
    E: Excellent support — completely consistent and well-grounded.

    Choose the best-fitting option:"""
]


class Verifier(object):
    def __init__(self, args):
        super(Verifier, self).__init__(args)

    def get_prompts(self, token, questions, knowledges, answers, instruction_index=0):

        if not isinstance(questions, list):
            questions = [questions]
            knowledges = [knowledges]
            answers = [answers]

        if token == "isRET":
            instructions_full = [
                INSTRUCTIONS_isRET[instruction_index].format(
                    question = question
                ) for question in questions
        ]

        elif token == "isREL":
            instructions_full = [
                INSTRUCTIONS_isREL[instruction_index].format(
                    question = question,
                    passage = knowledge
                ) for (question, knowledge) in zip(questions, knowledges)
        ]
            
        elif token == "isGRD":

            instructions_full = [
                INSTRUCTIONS_isGRD[instruction_index].format(
                    question=question,
                    passage=knowledge,
                    answer=answer
                )
                for question, knowledge, answer in zip(questions, knowledges, answers)
            ]

        else:
            print("error")

        return instructions_full
