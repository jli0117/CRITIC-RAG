# Vanilla
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval False --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name vanilla

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/flan-t5-base/vanilla/epoch_0_llm_answers.jsonl" 

# RAG
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval True --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name retrieve_topk

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/flan-t5-base/retrieve_topk/epoch_0_llm_answers.jsonl" 

# CRITIC-RAG
python main_verify_open_ended.py --debug False --data_name LiveQA --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval True --use_verification True \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name verification

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/flan-t5-base/verification/epoch_0_llm_answers.jsonl" 



# Vanilla
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval False --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name vanilla

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/gpt4/vanilla/epoch_0_llm_answers.jsonl" 

# RAG
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval True --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name retrieve_topk

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/gpt4/retrieve_topk/epoch_0_llm_answers.jsonl" 

# CRITIC-RAG
python main_verify_open_ended.py --debug False --data_name LiveQA --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval True --use_verification True \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name verification

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/gpt4/verification/epoch_0_llm_answers.jsonl" 


# Vanilla
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval False --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name vanilla

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/deepseek/vanilla/epoch_0_llm_answers.jsonl" 

# RAG
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval True --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name retrieve_topk

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/deepseek/retrieve_topk/epoch_0_llm_answers.jsonl" 

# CRITIC-RAG
python main_verify_open_ended.py --debug False --data_name LiveQA --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval True --use_verification True \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name verification

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/deepseek/verification/epoch_0_llm_answers.jsonl" 


# Vanilla
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval False --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name vanilla

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/medgemma/vanilla/epoch_0_llm_answers.jsonl" 

# RAG
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval True --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name retrieve_topk

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/medgemma/retrieve_topk/epoch_0_llm_answers.jsonl" 

# CRITIC-RAG
python main_verify_open_ended.py --debug False --data_name LiveQA --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval True --use_verification True \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name verification

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/medgemma/verification/epoch_0_llm_answers.jsonl" 


# Vanilla
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval False --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name vanilla_

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/Llama-3.2-3B/vanilla_/epoch_0_llm_answers.jsonl" 

# RAG
python main_rag_open_ended.py --debug False --data_name LiveQA --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval True --use_verification False \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name retrieve_topk

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/Llama-3.2-3B/retrieve_topk/epoch_0_llm_answers.jsonl" 

# CRITIC-RAG
python main_verify_open_ended.py --debug False --data_name LiveQA --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval True --use_verification True \
    --question_prefix "You are a medical expert. Please give long answer to the following question:" \
    --batch_size 32 --exp_name verification

python evaluation_long.py --data_name "LiveQA" \
    --results_file "./results/LiveQA/Llama-3.2-3B/verification/epoch_0_llm_answers.jsonl" 
