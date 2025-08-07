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
from models.verifiers import Verifier
from models.rewards import generate_label

from biorag import rr_biorag, kmeans_sampling

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
        self.verifier = VERIFIER_NAMES[glb_args.knowledge_base](ver_args)
        self.verifier_dataset = None

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
        prompt = f"{sample['question']}\n"
        prompt += "Options:\n"
        for key, value in sample['options'].items():
            prompt += f"{key}. {value}\n"
        return prompt


    def retrieve_only(self, samples):
        knowledges_list = []
        knowledges_ids_list = []

        questions = [self.combine_q_o(sample) for sample in samples]
        retrieved_knowledges, retrieved_knowledges_ids, retrieved_sources, retrieved_scores = rr_biorag.run(samples)
        knowledges_list.extend(retrieved_knowledges)
        knowledges_ids_list.extend(retrieved_knowledges_ids)
        print("Retrieval is used.")

        #return questions, knowledges_list, knowledges_ids_list
        return questions, retrieved_knowledges, retrieved_knowledges_ids, retrieved_sources, retrieved_scores



    def generate_only(self, samples, knowledges, combine=True):
        if combine:
            questions = [self.combine_q_o(sample) for sample in samples]
        else:
            questions = samples        
        answers = self.lang_model.generate(questions, knowledges, self.glb_args.knowledge_base)
        return questions, knowledges, answers


    def verify(self, token, questions, knowledges=[], answers=[], instruction_sets=[0]):
        predictions = []
        for index in instruction_sets:
            prediction = self.verifier.verify(token, questions, knowledges, answers, instruction_index=index)
            predictions.append(prediction)
        
        pred_all_probs = np.array([prediction[1] for prediction in predictions])[:, :3, :]
        pred_probs = np.sum(pred_all_probs, axis=0)
        pred_labels = np.argmax(pred_probs, 0)

        return pred_all_probs, pred_probs, pred_labels


    def eval(self):
        results = {
            'dataset_questions': [],
            'llm_answers': [],
            'dataset_correct_options': [],
            'isRET_token': [],
            'retrieved_evidences': [],
            'retrieved_sources': [],
            'retrieved_scores': [],
            'filtered_evidences': [],
            'filtered_sources': [],
            'filtered_scores': [],
            'final_evidences': [],
            'final_sources': [],
            'final_scores': [],
        }

        for index, batch in enumerate(tqdm(self.qa_data_loaders['test'])):
            if self.glb_args.debug and index == 1:  
                break
            
            isRET = [] 
            for sample in batch:
                instruction_index = random.randint(0, self.ver_args.verifier_num_instructions-1) if self.ver_args.ensemble else 0
                question = self.combine_q_o(sample)
                prompt_isRET = self.verifier.get_prompts('isRET', [question], [[]], [], instruction_index)[0]
                gpt_label = generate_label(prompt_isRET, "")
                isRET.append(gpt_label)

            # Split the batch based on whether the retrieval is needed
            batch_noret = [] 
            batch_ret = [] 
            
            for sample, ret_label in zip(batch, isRET):
                if ret_label == 'B':
                    batch_noret.append(sample)
                elif ret_label == 'A':
                    batch_ret.append(sample)

            # Initialize storage
            all_questions_noret, all_knowledges_noret, all_answers_noret, all_correct_noret = [], [], [], []
            all_questions_ret, all_knowledges_ret, all_answers_ret, all_correct_ret = [], [], [], []
            all_isRET_noret, all_isRET_ret = [], []

            # Process non-retrieval samples
            if batch_noret:
                batch_questions, _, batch_answers = self.generate_only(batch_noret, [[]]*len(batch_noret))
                all_questions_noret.extend(batch_questions)
                all_knowledges_noret.extend([[]]*len(batch_questions))
                all_answers_noret.extend(batch_answers)
                all_correct_noret.extend([sample['correct option'] for sample in batch_noret])
                all_isRET_noret.extend(['no'] * len(batch_noret))
                # save empty RAG information
                results['retrieved_evidences'].extend([[] for _ in batch_noret])
                results['retrieved_sources'].extend([[] for _ in batch_noret])
                results['retrieved_scores'].extend([[] for _ in batch_noret])
                results['filtered_evidences'].extend([[] for _ in batch_noret])
                results['filtered_sources'].extend([[] for _ in batch_noret])
                results['filtered_scores'].extend([[] for _ in batch_noret])
                results['final_evidences'].extend([[] for _ in batch_noret])
                results['final_sources'].extend([[] for _ in batch_noret])
                results['final_scores'].extend([[] for _ in batch_noret])                
                
        
            # Stage 2: Process samples that need retrieval
            if batch_ret:
                batch_questions, raw_knowledges, raw_knowledges_ids, raw_sources, raw_scores = self.retrieve_only(batch_ret)
                
                # save raw retrieved evidences
                results['retrieved_evidences'].extend(raw_knowledges)
                results['retrieved_sources'].extend(raw_sources)
                results['retrieved_scores'].extend([
                    [float(s) if isinstance(s, torch.Tensor) else s for s in score_list]
                    for score_list in raw_scores
                ])
                
                ret_correct = [sample['correct option'] for sample in batch_ret]  # Correct answers for retrieval samples

                filtered_questions, filtered_knowledges, filtered_correct = [], [], []

                for q, k, c, src_list, score_list in zip(batch_questions, raw_knowledges, ret_correct, raw_sources, raw_scores):
                    filtered_k = [] # knowledges
                    filtered_src = [] # sources
                    filtered_sco = [] # scores

                    # Filter evidences
                    for idx, evidence in enumerate(k):
                        instruction_index = random.randint(0, self.ver_args.verifier_num_instructions-1) if self.ver_args.ensemble else 0
                        prompt_isREL = self.verifier.get_prompts('isREL', [q], [[evidence]], [], instruction_index)[0]
                        gpt_label = generate_label(prompt_isREL, "")
                        
                        if gpt_label == 'B':
                            filtered_k.append(evidence)
                            filtered_src.append(src_list[idx])
                            score = score_list[idx]
                            filtered_sco.append(float(score) if isinstance(score, torch.Tensor) else score)

                    print("finish screening the evidences: ", len(filtered_k))
                    results['filtered_evidences'].append(filtered_k)
                    results['filtered_sources'].append(filtered_src)
                    results['filtered_scores'].append(filtered_sco)


                    if filtered_k:
                        if len(filtered_k) <= self.ver_args.verifier_num_clusters:
                            
                            filtered_questions.append(q)
                            filtered_knowledges.append(filtered_k)
                            filtered_correct.append(c)
                            print("finish save evidences less than 2")
                            results['final_evidences'].append(filtered_k)
                            results['final_sources'].append(filtered_src)
                            results['final_scores'].append(filtered_sco)

                            
                        else:
                            # Cluster sampling
                            sampled_docs = kmeans_sampling.multi_perspective_sampling(
                                self.ver_args.verifier_num_clusters, 
                                filtered_k, 
                                self.ver_args.verifier_num_repeat
                            )
                            
                            best_score = -1
                            best_answer = None
                            best_docs = None
                            
                            for docs in sampled_docs:
                                # Generate answer for this evidence set
                                _, _, sampled_answer = self.generate_only([q], [docs], combine=False)
                                answer = sampled_answer[0]
                                
                                # Get GPT rating
                                instruction_index = random.randint(0, self.ver_args.verifier_num_instructions-1) if self.ver_args.ensemble else 0
                                prompt_isGRD = self.verifier.get_prompts('isGRD', [q], [[docs]], [[answer]], instruction_index)[0]
                                
                                try:
                                    gpt_score_letter = generate_label(prompt_isGRD, "")  # Assuming score is numeric
                                    score_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
                                    gpt_score = score_mapping.get(gpt_score_letter, 0) 
                                except:
                                    gpt_score = 0  # Default score if parsing fails

                                # Track best result
                                if gpt_score > best_score:
                                    best_score = gpt_score
                                    best_answer = answer
                                    best_docs = docs

                            all_questions_ret.append(q)
                            all_knowledges_ret.append(best_docs)
                            all_answers_ret.append(best_answer)
                            all_correct_ret.append(c)
                            results['final_evidences'].append(best_docs)
                            results['final_sources'].append([filtered_src[filtered_k.index(d)] for d in best_docs])
                            results['final_scores'].append([filtered_sco[filtered_k.index(d)] for d in best_docs])

                    else:
                        filtered_questions.append(q)
                        filtered_knowledges.append([])
                        filtered_correct.append(c)
                        results['final_evidences'].append([])
                        results['final_sources'].append([])
                        results['final_scores'].append([])

                # Generate answers for filtered questions
                _, _, filtered_answers = self.generate_only(filtered_questions, filtered_knowledges, combine=False)

                all_questions_ret.extend(filtered_questions)
                all_knowledges_ret.extend(filtered_knowledges)
                all_answers_ret.extend(filtered_answers)
                all_correct_ret.extend(filtered_correct)

                # save retrieve labels
                all_isRET_ret.extend(['yes'] * len(all_questions_ret))

            # Store results
            results['dataset_questions'].extend(all_questions_noret + all_questions_ret)
            results['llm_answers'].extend(all_answers_noret + all_answers_ret)
            results['dataset_correct_options'].extend(all_correct_noret + all_correct_ret)
            results['isRET_token'].extend(all_isRET_noret + all_isRET_ret)

        return results


    def run(self):
        all_results = []

        for epoch in tqdm(range(self.ver_args.verifier_num_epochs)):
            if not self.ver_args.use_verification and epoch == 1:
                break

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

        with open(save_path, "w", encoding="utf-8") as file:
            for i in range(len(eval_result['llm_answers'])):
                combined_record = {
                    "llm_answer": eval_result['llm_answers'][i],
                    "correct_option": eval_result['dataset_correct_options'][i],
                    "question": eval_result['dataset_questions'][i],
                    "retrieved_evidences": eval_result['retrieved_evidences'][i],
                    "retrieved_sources": eval_result['retrieved_sources'][i],
                    "retrieved_scores": [float(s) for s in eval_result['retrieved_scores'][i]],
                    "filtered_evidences": eval_result['filtered_evidences'][i],
                    "filtered_sources": eval_result['filtered_sources'][i],
                    "filtered_scores": [float(s) for s in eval_result['filtered_scores'][i]],
                    "final_evidences": eval_result['final_evidences'][i],
                    "final_sources": eval_result['final_sources'][i],
                    "final_scores": [float(s) for s in eval_result['final_scores'][i]],
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
