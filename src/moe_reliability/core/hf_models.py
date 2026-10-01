###
# hf_models.py
#
# Loading pretrained models from Huggingface. Used to read and rewrite
# checkpoints; experiments are served with vLLM, and routed experts are read
# from the serving stack rather than from a separate Hugging Face forward pass.
# Dylan Everingham
# 18.02.2026
###

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Hugging Face access token for gated models (read from the environment)
hf_token = os.environ.get("HF_TOKEN")

def load_model(model_id, max_memory=None, enable_bnb=False):

    # Configure 4-bit quantization
    if enable_bnb:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.float16,
        trust_remote_code=False,
        quantization_config=quantization_config,
        token=hf_token,
    )

    tokenizer = load_tokenizer(model_id)
    return model, tokenizer

def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"load_tokenizer: no pad token defined for {model_id}, using eos_token ({tokenizer.eos_token!r}) as pad_token.")
    tokenizer.padding_side = "left"
    return tokenizer
