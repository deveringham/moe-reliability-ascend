###
# hf_models.py
#
# Routines for MoE experiments on pretrained models from Huggingface:
# model loading, chat generation and router activation recording.
# Dylan Everingham
# 18.02.2026
###

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .data import format_prompts_mmlu

# Hugging Face access token for gated models (read from the environment)
hf_token = os.environ.get("HF_TOKEN")

# torch device
device = torch.device("cuda")

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

# Function to get mask for valid activations / probabilities after batched
# activation recording. Masks out prompt padding and post-eos padding
def get_valid_activation_mask(attention_mask, generated_ids, tokenizer):
    
    # attention_mask [batch, padded_prompt_len]
    # generated_ids [batch, generation_steps]
    # Returns [batch, padded_prompt_len + generation_steps]
    
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids)
 
    batch_size, gen_len = generated_ids.shape
    step_len = max(gen_len - 1, 0)
    step_ids = generated_ids[:, :step_len]

    gen_mask = torch.ones((batch_size, step_len), dtype=torch.bool)
    for i in range(batch_size):
        for j in range(step_len):
            if step_ids[i, j].item() in eos_ids:
                # Keep the EOS step itself (a genuine forward pass); drop
                # everything generated after it for this sample.
                gen_mask[i, j + 1:] = False
                break
 
    prompt_mask = attention_mask.bool().cpu()
    return torch.cat([prompt_mask, gen_mask], dim=1)

def chat_generate(model, tokenizer, probe, prompt="", max_new_tokens=100, clear_probe=True, prompt_formatted=True):

    # Clear monitoring probe
    if clear_probe:
        probe.clear()
        
    # Set chat template and transfer to device
    if prompt_formatted:
        messages = prompt
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(
        model_inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        attention_mask=model_inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,

    )
    generated_ids = torch.stack([
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ])

    # Decode
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Get metrics
    probs = probe.get_probs()
    active_experts = probe.get_active_experts()
    
    return response, probs, active_experts
        

def chat_generate_batched(model, tokenizer, probe, prompts=[""], max_new_tokens=100, clear_probe=True, prompt_formatted=True, max_length=1024):
 
    # Clear monitoring probe
    if clear_probe:
        probe.clear()
 
    # Apply chat template to a list of prompts
    texts = []
    for prompt in prompts:
        if prompt_formatted:
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        
    # Tokenize with padding
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

    # Generate output
    attention_mask=model_inputs['attention_mask']
    generated_ids = model.generate(
        input_ids=model_inputs['input_ids'],
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids = torch.stack([
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ])

    # Decode
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Get mask representing valid activation data i.e. not padding
    valid_mask = get_valid_activation_mask(attention_mask, generated_ids, tokenizer)
 
    batch_size = len(prompts)
    probs = probe.get_probs(batch_size=batch_size)
    active_experts = probe.get_active_experts(batch_size=batch_size)

    return responses, probs, active_experts, valid_mask

# Function to right-pad a tensor's seq dim (dim=1) up to target length
def _pad_seq(tensor, target_len, pad_value=0):
   
    cur_len = tensor.shape[1]
    if cur_len == target_len:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[1] = target_len - cur_len
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad_tensor], dim=1)
 

# Function to merge outputs of two calls to chat_generte with different batch lengths
# used in chat_generate_batched_with_oom_retry
def _merge_batch_results(results):

    all_responses = []
    for r in results:
        all_responses.extend(r[0])
 
    max_seq_len = max(r[1].shape[1] for r in results)
 
    probs_list = [_pad_seq(r[1], max_seq_len, pad_value=0.0) for r in results]
    active_experts_list = [_pad_seq(r[2], max_seq_len, pad_value=0) for r in results]
    valid_mask_list = [_pad_seq(r[3], max_seq_len, pad_value=False) for r in results]
 
    merged_probs = torch.cat(probs_list, dim=0)
    merged_active_experts = torch.cat(active_experts_list, dim=0)
    merged_valid_mask = torch.cat(valid_mask_list, dim=0)
 
    return all_responses, merged_probs, merged_active_experts, merged_valid_mask
 
 
def chat_generate_batched_with_oom_retry(model, tokenizer, probe,
                                         prompts=[""], max_new_tokens=100, clear_probe=True,
                                         prompt_formatted=True, max_length=1024, min_batch=1):
    
    try:
        return chat_generate_batched(
            model, tokenizer, probe, prompts=prompts,
            max_new_tokens=max_new_tokens, clear_probe=clear_probe,
            prompt_formatted=True, max_length=max_length
        )
    except torch.cuda.OutOfMemoryError:
        pass

    torch.cuda.empty_cache()

    if len(prompts) <= min_batch:
        # Can't split further
        raise

    mid = len(prompts) // 2
    print(f"generate_with_oom_retry: OOM on batch of {len(prompts)}, "
          f"retrying as sub-batches of {mid} and {len(prompts) - mid}...")

    first_half = prompts[:mid]
    second_half = prompts[mid:]

    result_1 = chat_generate_batched_with_oom_retry(model, tokenizer, probe,
                                       prompts=first_half, max_new_tokens=max_new_tokens,
                                       clear_probe=clear_probe, prompt_formatted=prompt_formatted,
                                       max_length=max_length, min_batch=min_batch)
    result_2 = chat_generate_batched_with_oom_retry(model, tokenizer, probe,
                                       prompts=second_half, max_new_tokens=max_new_tokens,
                                       clear_probe=clear_probe, prompt_formatted=prompt_formatted,
                                       max_length=max_length, min_batch=min_batch)

    results = _merge_batch_results([result_1, result_2])
    return results
        

def get_activations_mmlu(model, tokenizer, dataset, probe, batch_size=1, max_new_tokens=100):
    
    # Results will contain router logits for each sample plus prompt and response
    results = []

    # Formatted prompts
    prompts, subjects, questions = format_prompts_mmlu(dataset, prompt_reps=1)
    n_samples = len(prompts)
    
    # For each sample...
    if batch_size > 1:
        
        # Batched version
        for batch_start in range(0, n_samples, batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
     
            print(f"Generating responses {batch_start + 1}-{batch_start + len(batch_prompts)}/{n_samples}...")
     
            # Start timing
            start_time = time.perf_counter()
     
            # Generate batch of responses and get metrics + per-sample validity mask
            # If OOM occurs, repeat with half batch_size
            responses, probs, active_experts, valid_mask = chat_generate_batched_with_oom_retry(
                model, tokenizer, probe, prompts=batch_prompts,
                max_new_tokens=max_new_tokens, prompt_formatted=True)
     
            # Stop timing
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            per_sample_time = inference_time / len(batch_prompts)
            print(f"Batch inference took {inference_time:.3f}s ({per_sample_time:.3f}s/sample).")
     
            # Split the batch back into individual sample results, keeping only
            # the valid (non-padding, pre/at-EOS) positions of each sample's routing data
            for i in range(len(batch_prompts)):
                count = i + batch_start
     
                result = {}
                sample_mask = valid_mask[i] # [seq_len]
                result['idx'] = count
                result['probs'] = probs[i][sample_mask] # [valid_len, n_experts, n_routers]
                result['active_experts'] = active_experts[i][sample_mask] # [valid_len, k, n_routers]
                result['prompt'] = batch_prompts[i] #questions[count]
                result['response'] = responses[i]
                #text = tokenizer.apply_chat_template(prompts[count], tokenize=False, add_generation_prompt=True)
                #result['prompt_tokenized'] = tokenizer([text])
                #result['response_tokenized'] = tokenizer.encode(responses[i])
                result['subject'] = subjects[count]
                result['inference_time'] = per_sample_time
                
                results.append(result)

            # Clear memory usage between batches
            torch.cuda.empty_cache()
                
    else:
        count = 0
        for d in dataset:
        
            print(f"Generating response {count+1}/{n_samples}...")
            
            # Start timing
            start_time = time.perf_counter()
    
            # Generate response and get metrics
            response, probs, active_experts = chat_generate(model, tokenizer, probe,
                                                            prompt=prompts[count], max_new_tokens=max_new_tokens,
                                                            prompt_formatted=True)
            
            # Stop timing
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.3f}s.")
    
            # Store results
            result = {}
            result['probs'] = probs.squeeze()
            result['active_experts'] = active_experts.squeeze()       
            result['prompt'] = d['question']
            result['response'] = response
            text = tokenizer.apply_chat_template(prompts[count], tokenize=False, add_generation_prompt=True)
            result['prompt_tokenized'] = tokenizer([text])
            result['response_tokenized'] = tokenizer.encode(response)
            result['subject'] = d['subject']
            result['inference_time'] = inference_time
            
            results.append(result)
    
            count += 1

    return results
