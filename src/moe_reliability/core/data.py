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

from datasets import load_dataset


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


# --- Workload families -------------------------------------------------------
#
# Routing depends on what a prompt is about, so a detector's benign floor has to
# be measured across more than one kind of traffic. Each family yields chat
# messages and a category, which becomes the record's ``subject`` as
# "<family>/<category>".
#
# Unlike get_data_mmlu, which takes the first N rows and then shuffles (so 3000
# samples cover only the alphabetically first 19 subjects), these shuffle the
# stream before taking, so a sample spans the dataset. get_data_mmlu is left as
# it is because existing workloads are keyed to its ordering.

WORKLOAD_FAMILIES = ("mmlu", "gsm8k", "mbpp", "mmmlu", "dolly", "ultrachat")

_SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def _stream(dataset_id, config, split, n, seed):
    ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
    return list(ds.shuffle(seed=seed, buffer_size=10000).take(n))


def _mmlu_style(question, choices):
    return (f"The following is a multiple-choice question.\nQuestion: {question}\n"
            + "".join(f"{label}) {c}\n" for label, c in zip("ABCD", choices))
            + "\nThink step-by-step, explaining your reasoning before giving the final answer.")


def _family_prompts(family, arg, n, seed):
    """[(messages, category)] for one family."""
    if family == "mmlu":
        rows = _stream("cais/mmlu", "all", "test", n, seed)
        return [([_SYSTEM, {"role": "user", "content": _mmlu_style(r["question"], r["choices"])}], r["subject"])
                for r in rows]
    if family == "gsm8k":
        rows = _stream("openai/gsm8k", "main", "test", n, seed)
        return [([_SYSTEM, {"role": "user", "content": f"{r['question']}\nSolve this step by step."}], "math")
                for r in rows]
    if family == "mbpp":
        rows = _stream("google-research-datasets/mbpp", "full", "test", n, seed)
        return [([_SYSTEM, {"role": "user", "content": f"{r['text']}\nYour code should pass this test:\n"
                                                         f"{r['test_list'][0]}"}], "python")
                for r in rows]
    if family == "mmmlu":
        lang = arg or "DE_DE"
        rows = _stream("openai/MMMLU", lang, "test", n, seed)
        # The question and options are translated; the instruction stays in English,
        # as a multilingual deployment's system prompts typically would.
        return [([_SYSTEM, {"role": "user", "content": _mmlu_style(r["Question"], [r[c] for c in "ABCD"])}],
                 f"{lang}:{r['Subject']}") for r in rows]
    if family == "dolly":
        rows = _stream("databricks/databricks-dolly-15k", "default", "train", n, seed)
        return [([_SYSTEM, {"role": "user", "content": (f"{r['context']}\n\n" if r["context"] else "")
                                                         + r["instruction"]}], r["category"]) for r in rows]
    if family == "ultrachat":
        rows = _stream("HuggingFaceH4/ultrachat_200k", "default", "test_sft", n, seed)
        return [([_SYSTEM, {"role": "user", "content": r["prompt"]}], "chat") for r in rows]
    raise ValueError(f"unknown workload family {family!r}; expected one of {WORKLOAD_FAMILIES}")


def parse_workload(spec):
    """``"gsm8k"``, ``"mmmlu:ZH_CN"`` or ``"mixed:mmlu,gsm8k,mmmlu:ZH_CN"`` -> [(family, arg)]."""
    spec = spec.strip()
    parts = spec[len("mixed:"):].split(",") if spec.startswith("mixed:") else [spec]
    out = []
    for part in parts:
        family, _, arg = part.strip().partition(":")
        if family not in WORKLOAD_FAMILIES:
            raise ValueError(f"workload {spec!r}: unknown family {family!r}; expected one of {WORKLOAD_FAMILIES}")
        if arg and family != "mmmlu":
            raise ValueError(f"workload {spec!r}: only mmmlu takes an argument (a language, e.g. mmmlu:ZH_CN)")
        out.append((family, arg or None))
    if not out:
        raise ValueError(f"workload {spec!r} names no family")
    return out


def workload_prompts(spec, n, seed):
    """Prompts and their "<family>/<category>" labels for a workload spec.

    A mixed spec splits n as evenly as possible between its families and
    interleaves them, so a window of consecutive prompts holds every family.
    """
    families = parse_workload(spec)
    per = [n // len(families) + (i < n % len(families)) for i in range(len(families))]
    blocks = []
    for (family, arg), k in zip(families, per):
        name = family if arg is None else f"{family}:{arg}"
        blocks.append([(m, f"{name}/{c}") for m, c in _family_prompts(family, arg, k, seed)])
    mixed = [item for group in zip(*blocks) for item in group]
    longest = max(len(b) for b in blocks)
    mixed += [b[i] for i in range(min(len(b) for b in blocks), longest) for b in blocks if i < len(b)]
    return [m for m, _ in mixed], [c for _, c in mixed]
