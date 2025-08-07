# Vanilla
python main_rag_options.py --data_name MMLU_merged --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval False --use_verification False \
    --batch_size 32 --exp_name vanilla

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/flan-t5-base/vanilla/epoch_0_llm_answers.jsonl" \
    --model_type "flan"


# RAG
python main_rag_options.py --debug False --data_name MMLU_merged --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval True --use_verification False \
    --batch_size 32 --exp_name retrieve_topk

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/flan-t5-base/retrieve_topk/epoch_0_llm_answers.jsonl" \
    --model_type "flan"


# CRITIC-RAG
python main_verify_options.py --debug False --data_name MMLU_merged --model_type flan \
    --model_name_or_path ../../tmp/flan-t5-base --use_retrieval True --use_verification True \
    --batch_size 32 --exp_name verification

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/flan-t5-base/verification/epoch_0_llm_answers.jsonl" \
    --model_type "flan"



# Vanilla
python main_rag_options.py --data_name MMLU_merged --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval False --use_verification False \
    --batch_size 32 --exp_name vanilla

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/gpt4/vanilla/epoch_0_llm_answers.jsonl" \
    --model_type "gpt4"


# RAG
python main_rag_options.py --debug False --data_name MMLU_merged --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval True --use_verification False \
    --batch_size 32 --exp_name retrieve_topk

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/gpt4/retrieve_topk/epoch_0_llm_answers.jsonl" \
    --model_type "gpt4"


# CRITIC-RAG
python main_verify_options.py --debug False --data_name MMLU_merged --model_type gpt4 \
    --model_name_or_path ../../tmp/gpt4 --use_retrieval True --use_verification True \
    --batch_size 32 --exp_name verification

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/gpt4/verification/epoch_0_llm_answers.jsonl" \
    --model_type "gpt4"



# Vanilla
python main_rag_options.py --data_name MMLU_merged --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval False --use_verification False \
    --batch_size 32 --exp_name vanilla

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/deepseek/vanilla/epoch_0_llm_answers.jsonl" \
    --model_type "deepseek"

# RAG
python main_rag_options.py --debug False --data_name MMLU_merged --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval True --use_verification False \
    --batch_size 32 --exp_name retrieve_topk

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/deepseek/retrieve_topk/epoch_0_llm_answers.jsonl" \
    --model_type "deepseek"

# CRITIC-RAG
python main_verify_options.py --debug False --data_name MMLU_merged --model_type deepseek \
    --model_name_or_path ../../tmp/deepseek --use_retrieval True --use_verification True \
    --batch_size 32 --exp_name verification

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/deepseek/verification/epoch_0_llm_answers.jsonl" \
    --model_type "deepseek"



# Vanilla
python main_rag_options.py --data_name MMLU_merged --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval False --use_verification False \
    --batch_size 32 --exp_name vanilla

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/medgemma/vanilla/epoch_0_llm_answers.jsonl" \
    --model_type "medgemma"

# RAG
python main_rag_options.py --debug False --data_name MMLU_merged --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval True --use_verification False \
    --batch_size 32 --exp_name retrieve_topk

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/medgemma/retrieve_topk/epoch_0_llm_answers.jsonl" \
    --model_type "medgemma"

# CRITIC-RAG
python main_verify_options.py --debug False --data_name MMLU_merged --model_type medgemma \
    --model_name_or_path ../../tmp/medgemma --use_retrieval True --use_verification True \
    --batch_size 32 --exp_name verification

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/medgemma/verification/epoch_0_llm_answers.jsonl" \
    --model_type "medgemma"


# Vanilla
python main_rag_options.py --data_name MMLU_merged --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval False --use_verification False \
    --batch_size 32 --exp_name vanilla

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/Llama-3.2-3B/vanilla/epoch_0_llm_answers.jsonl" \
    --model_type "llama3"


# RAG
python main_rag_options.py --debug False --data_name MMLU_merged --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval True --use_verification False \
    --batch_size 32 --exp_name retrieve_topk

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/Llama-3.2-3B/retrieve_topk/epoch_0_llm_answers.jsonl" \
    --model_type "llama3"


# CRITIC-RAG
python main_verify_options.py --debug False --data_name MMLU_merged --model_type llama3 \
    --model_name_or_path ../../tmp/Llama-3.2-3B --use_retrieval True --use_verification True \
    --batch_size 32 --exp_name verification

python evaluation.py --data_name "MMLU_merged" \
    --results_file "./results/MMLU_merged/Llama-3.2-3B/verification/epoch_0_llm_answers.jsonl" \
    --model_type "llama3"