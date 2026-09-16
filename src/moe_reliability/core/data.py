###
# data.py
#
# Data loading and prompt formatting routines for MoE experiments.
# Dylan Everingham
# 02.02.2026
###

###
# 
#
# desc
# Dylan Everingham
# 16.09.2026
###

from datasets import load_dataset, get_dataset_config_names


def get_mmlu_subjects():
    all_subsets = get_dataset_config_names("cais/mmlu")
    subjects = [s for s in all_subsets if s not in ["all", "auxiliary_train"]]
    return subjects

def get_data_mmlu(n_samples=100, shuffle_seed=100, subset="all"):
    
    data_config = {
        "dataset_id": "cais/mmlu",
        "subset": subset,
        "context_length": 128,
        "shuffle_buffer": 10000,
        "n_samples": n_samples,
    }
    
    print(f"Streaming {data_config['dataset_id']} ({data_config['subset']}) (samples: {data_config['n_samples']})...")
    
    # Load dataset in streaming mode
    dataset = load_dataset(
        data_config["dataset_id"], 
        name=data_config["subset"],
        split="test", 
        streaming=True
    )
    
    # Take a small sample of the data
    dataset = dataset.take(data_config["n_samples"])

    # Shuffle
    dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=data_config["shuffle_buffer"])
    
    dataset = dataset.with_format("torch")
    return dataset

def format_prompts_mmlu(dataset, prompt_reps=1):
    
    messages_list = []
    subjects = []
    questions = []
    for d in dataset:
        messages = [
            {
                "role": "system", 
                "content": "You are a logical reasoning assistant. You must provide all of your reasoning, explanations, and final answers entirely in English. Do not use any other language."
            },
            {
                "role": "user", 
                "content": (
                    f"The following is a multiple-choice question.\n"
                    f"Question: {d['question']}\n"
                    f"A) {d['choices'][0]}\nB) {d['choices'][1]}\nC) {d['choices'][2]}\nD) {d['choices'][3]}\n\n"
                    f"Do not simply output the letter. Think step-by-step, carefully explaining your "
                    f"reasoning for each option before arriving at the final answer. Your entire response must be strictly in English."
                )
            }
        ]
        for _ in range(prompt_reps):
            messages_list.append(messages)
            subjects.append(d['subject'])
            questions.append(d['question'])
        
    return messages_list, subjects, questions
