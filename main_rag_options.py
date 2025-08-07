import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import pickle
import logging
import pdb
from tqdm import tqdm
import numpy as np
import torch
import json
import itertools

from transformers import HfArgumentParser, set_seed

from configs import GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments, VerifierArguments
from data import get_qa_datasets
from models.language_models import LanguageModel
from models.retrievers import Retriever

from biorag import rr_biorag

class Runner(object):
    def __init__(self, glb_args, data_args, lm_args, ret_args, ver_args):
        super(Runner, self).__init__()

        self.glb_args = glb_args
        self.data_args = data_args
        self.lm_args = lm_args
        self.ret_args = ret_args
        self.ver_args = ver_args

        self.output_path = self.get_output_path()
        self.logger = self.get_logger()
        self.logger.info(f"Global Arguments: {glb_args}")
        self.logger.info(f"Dataset Arguments: {data_args}")
        self.logger.info(f"Language Model Arguments: {lm_args}")
        self.logger.info(f"Retriever Arguments: {ret_args}")
        self.logger.info(f"Verifier Arguments: {ver_args}")

        set_seed(glb_args.seed)

        self.qa_datasets, self.qa_data_loaders = get_qa_datasets(data_args)
        self.lang_model = LanguageModel(lm_args)
        self.retriever = RETRIEVER_NAMES[glb_args.knowledge_base](ret_args)


    def get_logger(self):
        logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt = "%m/%d/%Y %H:%M:%S",
            level = logging.INFO,
            filename = f"{self.output_path}/logs.txt"
        )
        logger = logging.getLogger(__name__)
        return logger

    def combine_q_o(self, sample):
        if sample['options'] != '':
            prompt = f"{sample['question']}\n"
            prompt += "Options:\n"
            for key, value in sample['options'].items():
                prompt += f"{key}. {value}\n"
            return prompt
        else:
            return sample['question']

    def retrieve_only(self, samples):
        questions = [self.combine_q_o(sample) for sample in samples]

        if self.ret_args.use_retrieval:
            retrieved_knowledges, retrieved_knowledges_ids, retrieved_sources, retrieved_scores = rr_biorag.run(samples)
            print("Retrieval is used w/o verification")
        else:
            retrieved_knowledges = [[] for _ in samples]
            retrieved_knowledges_ids = [[] for _ in samples]
            retrieved_sources = [[] for _ in samples]
            retrieved_scores = [[] for _ in samples]
            print("Retrieval is NOT used")

        return questions, retrieved_knowledges, retrieved_knowledges_ids, retrieved_sources, retrieved_scores


    def generate_only(self, samples, knowledges, knowledges_ids):
        questions = [self.combine_q_o(sample) for sample in samples]        
        answers = self.lang_model.generate(questions, knowledges, self.glb_args.knowledge_base)
        return questions, knowledges, knowledges_ids, answers


    def eval(self):
        results = {
            'llm_answers': [],
            'dataset_questions': [],
            'dataset_options': [],
            'dataset_correct_answers': [],
            'dataset_correct_options': [],
            'rag_sources': [],                
            'rag_scores': [],                 
            'rag_evidences': []     
        }

        for index, batch in enumerate(tqdm(self.qa_data_loaders['test'])):
            if self.glb_args.debug and index == 1:
                break

            questions, knowledges, knowledges_ids, sources, scores = self.retrieve_only(batch)

            questions, knowledges, knowledges_ids, answers = self.generate_only(batch, knowledges, knowledges_ids)

            if self.ver_args.use_verification:
                print('Not Implemented')

            results['llm_answers'].extend(answers)
            results['dataset_questions'].append([element['question'] for element in batch])
            results['dataset_options'].append([element['options'] for element in batch])
            results['dataset_correct_options'].append([element['correct option'] for element in batch])
            results['rag_sources'].extend(sources)
            results['rag_scores'].extend(scores)
            results['rag_evidences'].extend(knowledges)

        return {
            'samples': self.qa_datasets['test'],
            'llm_answers': results['llm_answers'],
            'dataset_questions': results['dataset_questions'],
            'dataset_options': results['dataset_options'],
            'dataset_correct_options': results['dataset_correct_options'],
            'rag_evidences': results['rag_evidences'],
            'rag_sources': results['rag_sources'],
            'rag_scores': results['rag_scores']
        }


    def run(self):
        all_results = []

        for epoch in tqdm(range(self.ver_args.verifier_num_epochs)):
            if not self.ver_args.use_verification and epoch == 1:
                break

            if self.ver_args.use_verification:
                print('Not Implemented')

            with torch.no_grad():
                eval_result = self.eval()
                all_results.append(eval_result)
                self.save_llm_answers(epoch, eval_result)

        self.save_results(all_results[0])


    def get_output_path(self):
        output_path = f"./results/{self.data_args.data_name}/{self.lm_args.model_name_or_path.split('/')[-1]}/{self.glb_args.exp_name}"
        if not os.path.exists(output_path): os.makedirs(output_path)
        print(f"OUTPUT_PATH: {output_path}")
        return output_path

    def save_results(self, results):
        with open(f"{self.output_path}/results.pkl", "wb") as outfile:
            pickle.dump(results, outfile)

    def save_llm_answers(self, epoch, eval_result):
        results_dir = f"{self.output_path}/"
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"epoch_{epoch}_llm_answers.jsonl")

        all_correct_option = list(itertools.chain.from_iterable(eval_result["dataset_correct_options"]))
        all_question = list(itertools.chain.from_iterable(eval_result["dataset_questions"]))
        all_options = list(itertools.chain.from_iterable(eval_result["dataset_options"]))

        rag_evidences = eval_result["rag_evidences"]
        rag_sources = eval_result["rag_sources"]
        rag_scores = [[float(score) for score in sample_scores] for sample_scores in eval_result["rag_scores"]]

        with open(save_path, "w", encoding="utf-8") as file:
            for i in range(len(eval_result['llm_answers'])):
                combined_record = {
                    "llm_answer": eval_result['llm_answers'][i],
                    "correct_option": all_correct_option[i],
                    "question": all_question[i],
                    "options": all_options[i],
                    "evidences": rag_evidences[i],  
                    "sources": rag_sources[i],      
                    "scores": rag_scores[i]   
                }
                file.write(json.dumps(combined_record, ensure_ascii=False) + "\n")

        print(f"LLM answers for epoch {epoch} saved to {save_path}")




if __name__ == "__main__":

    parser = HfArgumentParser((GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments, VerifierArguments))

    glb_args, data_args, lm_args, ret_args, ver_args = parser.parse_args_into_dataclasses()

    glb_args.device = lm_args.device = ret_args.device = ver_args.device = torch.device("cuda" if torch.cuda.is_available() and not glb_args.no_cuda else "cpu")
    glb_args.n_gpu = lm_args.n_gpu = ret_args.n_gpu = ver_args.n_gpu = 0 if glb_args.no_cuda else torch.cuda.device_count()
    ver_args.cache_dir = ret_args.cache_dir = lm_args.cache_dir
    ver_args.device_map = ret_args.device_map = lm_args.device_map

    data_args.data_splits = ['train', 'test']

    runner = Runner(glb_args, data_args, lm_args, ret_args, ver_args)
    runner.run()
    if glb_args.stop:   import pdb; pdb.set_trace()
