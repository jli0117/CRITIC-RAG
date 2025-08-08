import torch
import pdb
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer
from openai import OpenAI
import concurrent.futures
from transformers import AutoProcessor, AutoModelForImageTextToText


class LanguageModel(object):
    def __init__(self, args):
        super(LanguageModel, self).__init__()

        self.args = args
        self.model = self.get_model()

    def get_config(self):
        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
        )
        return config

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
            resume_download = True
        )
        return tokenizer

    def get_model(self):
        if self.args.model_type == 'flan':
            
            self.config = self.get_config()

            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model_name_or_path,
                config = self.config,
                cache_dir = self.args.cache_dir,
                torch_dtype = torch.float16 if self.args.fp16 else None,
                device_map = self.args.device_map if self.args.n_gpu >= 0 else None,
                resume_download = True
            )
            return model
        

        elif self.args.model_type == 'llama3':
            pipeline = transformers.pipeline(
                "text-generation", self.args.model_name_or_path, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
            )
            return pipeline
  
        elif self.args.model_type == 'medgemma':
            model = AutoModelForImageTextToText.from_pretrained(
                self.args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            return model
        
        elif self.args.model_type == "gpt3":
            return None
        
        elif self.args.model_type == "gpt4":
            return None

        elif self.args.model_type == "deepseek":
            return None     



    def gpt_generate_answer(self, prompt, modelname):
        client = OpenAI(api_key="", base_url="")
        try:
            response = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": "You are an assistant in answering questions in medical domain"},
                    {"role": "user", "content": prompt}
                ]
            )
            gpt_raw_answer = response.choices[0].message.content
        except:
            raise NotImplementedError
        return gpt_raw_answer


    def batch_generate_answers(self, prompts, modelname, max_workers=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda p: self.gpt_generate_answer(p, modelname), prompts))
        return results
    

    def batch_deepseek_generate_answers(self, prompts, max_workers=5):
        from openai import OpenAI
        client = OpenAI(api_key="", base_url="")

        def call_api(prompt):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"[Error]: {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(call_api, prompts))

        return results

    
    def verbalize_knowledge(self, knowledge, knowledge_base, top1=False):
        if not knowledge: ## knowledge is a empty list
            return ''

        if top1:
            return f'Below is the passage meaningful to answer the question. \n{knowledge[0]} \n\n'
        else:
            return f'Below are passages meaningful to answer the question. \n{knowledge[:5]} \n\n'
            
        string = 'Below are facts in the form of the triple meaningful to answer the question. \n'
        for triplet in knowledge:
            string += f'({triplet[0]}, {triplet[1]}, {triplet[2]}) \n'
        string += '\n'
        return string


    def generate(self, questions, knowledges):
        prompts = [
            f'{self.verbalize_knowledge(knowledge)}{self.args.question_prefix}{question}\n{self.args.question_postfix}' \
            for (question, knowledge) in zip(questions, knowledges)
        ]

        if self.args.model_type == 'flan':

            self.tokenizer = self.get_tokenizer()

            input_tokens = self.tokenizer(
                prompts, max_length=self.args.max_source_length, padding='longest', truncation=True, return_tensors='pt',
            ).to(self.args.device)

            generated_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=self.args.max_target_length
            )

            outputs = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )
                    
        elif self.args.model_type == 'llama3':
            response = self.model(\
                prompts, max_new_tokens=self.args.max_target_length, \
                return_full_text=False)
            outputs = [res[0]['generated_text'] for res in response]

        
        elif self.args.model_type == 'medgemma':
            processor = AutoProcessor.from_pretrained(self.args.model_name_or_path)
            outputs = []
            for prompt in prompts:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    pad_to_multiple_of=8,
                ).to(self.model.device, dtype=torch.bfloat16)


                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
                    generation = generation[0][input_len:]

                output = processor.decode(generation, skip_special_tokens=True)
                outputs.append(output)
           
    
        elif self.args.model_type == 'gpt3':
            outputs = self.batch_generate_answers(prompts, modelname='gpt-3.5-turbo')

        elif self.args.model_type == 'gpt4':
            outputs = self.batch_generate_answers(prompts, modelname='gpt-4o')
            
        elif self.args.model_type == 'deepseek':
            outputs = self.batch_deepseek_generate_answers(prompts)
            
                        
    
        return outputs