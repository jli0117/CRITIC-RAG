from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "tmp/finetuned/llama3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

def generate_label(original_prompt, correct_answer):

    if correct_answer == "":
        input_data = f"{original_prompt}. You should ONLY give answer between A or B. DO NOT give any explanation."
    else:
        input_data = f"{original_prompt}.\n\nYou should know that the correct answer for the medical task is {correct_answer}. You should ONLY give answer between A or B. DO NOT give any explanation."

    try:
        inputs = tokenizer(input_data, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.7,
            top_p=0.9
        )
        gpt_raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        gpt_raw_answer = gpt_raw_answer[len(input_data):].strip()

        print(input_data)
        print("---------------------")
        print(gpt_raw_answer)

    except Exception as e:
        raise NotImplementedError(f"Error during inference: {e}")

    # Post-process the raw answer
    answer = gpt_raw_answer.split('.')[0].strip()

    if answer not in ['A', 'B', 'C', 'D', 'E']:
        print('Answer NOT Formatted')
        answer = 'A'

    return answer