###
# synthetic_workloads.py
#
# Generation of workloads with tunable load balance properties
# based on routing analysis.
# Dylan Everingham
# 17.08.2026
###

import torch
from tqdm import tqdm

# Get coefficient of variance
# scale-invariant, works for any normalization
def cvs_of(sums, dim):          
    return sums.std(dim=dim, unbiased=False) / sums.mean(dim=dim)
    
def get_qs(results, n_experts, n_layers, k, weighted_by_token_count=False):
    n_samples = len(results)
    qs = []
    total_tokens = 0
    
    for r in results:
        n_tokens = r['routed_experts'].shape[0] + r['prompt_routed_experts'].shape[0]
        active_experts = torch.cat([torch.tensor(r['routed_experts']), torch.tensor(r['prompt_routed_experts'])], dim=0) # (n_tokens, n_layers, k)
        active_experts = active_experts.swapdims(1,2) # (n_tokens, k, n_layers)
        active_experts_flat = torch.tensor(active_experts, dtype=torch.int32).flatten(start_dim=0, end_dim=1) # (n_tokens * k, n_layers)
        q = torch.zeros((n_experts, n_layers))
        ones = torch.ones_like(active_experts_flat, dtype=torch.float32)
        q.scatter_add_(dim=0, index=active_experts_flat, src=ones)
        if weighted_by_token_count:
            q = q/k
        else:
            q = q / (n_tokens*k)
        qs.append(q)
        total_tokens += n_tokens

    qs = torch.stack(qs, dim=0)
    
    return qs
    
    
# Constructs a workload for which the coefficient of variance of activation frequencies across
# experts in each layer (for both prefill and decode) is as close as possible to some requested values
#    results: activation recording results from get_results
#    l: desired number of tokens in workload, < #total_tokens
#    target_cvs: requested CVs (n_layers)
# Returns array of prompts from results of length l
#    and the obtained CVs (n_layers)
def construct_workload_cvs(results, qs, n_experts, n_layers, k, l, target_cvs, verbose=False, max_repeats=0):

    prompts = [r['prompt'] for r in results]
    n_samples = len(results)
    token_counts = [r['routed_experts'].shape[0] + r['prompt_routed_experts'].shape[0] for r in results]
    selected_indices = []
    
    # Keep track of the sum of the selected frequencies
    current_sum = torch.zeros_like(qs[0,:,:])
    
    # Keep track of selected prompts
    selected_mask = torch.zeros(n_samples, dtype=torch.int32)

    # Until we reach the desired number of tokens...
    n_current_tokens = 0
    with tqdm(total=100.0, disable=not verbose) as pbar:
        while n_current_tokens < l:
            
            # Calculate what the CV would be if we added each of the available prompts
            candidate_sums = current_sum.unsqueeze(0) + qs # (n_samples, n_experts, n_layers)
    
            # Compute CV per layer
            #candidate_cvs = candidate_std_across_experts * n_experts # (n_samples, n_layers)
            candidate_cvs = cvs_of(candidate_sums, dim=1)
            #print(candidate_cvs)
            
            # Calculate Mean Squared Error (or L2 distance) for each candidate
            distances = ((candidate_cvs - target_cvs.unsqueeze(0)) ** 2).sum(dim=1)
            
            # Set the distance of already selected indices to infinity so they aren't chosen again
            distances.masked_fill_(selected_mask>max_repeats, float('inf'))
            
            # Find the index with the minimum distance
            best_idx = torch.argmin(distances).item()
            
            # Update our trackers
            selected_indices.append(best_idx)
            selected_mask[best_idx] += 1
            current_sum += qs[best_idx, :, :]
            n_current_tokens += token_counts[best_idx]

            # Update progress bar
            pbar.update(100*token_counts[best_idx]/l)

    # Done if we have reached our token limit
    
    # Get final CVs and return
    obtained_cvs = cvs_of(current_sum, dim=0)
    selected_prompts = [prompts[i] for i in selected_indices]
    return selected_prompts, obtained_cvs, selected_indices

def evaluate_workload_quality_cvs(target_cvs, obtained_cvs):

    diff = obtained_cvs - target_cvs

    # Calculate evaluation metrics
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_error = diff.abs().max().item()
    
    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        # Add pmr
    }

def workload_sweep_cvs(results, qs, n_experts, n_layers, k, target_alphas, target_ls, cv_nat, max_repeats=0, verbose=False):
    
    workloads = {}
    
    if verbose:
        print(f'Generating synthetic workloads...')
    for l in target_ls:
        if verbose:
            print(f'Length: {l} tokens.')
        workloads[l] = {}
        for i, a in enumerate(target_alphas):
            if verbose:
                print(f'alpha: {a}')
            workload = {}
            target_cvs = cv_nat * a
            p, cvs, indices = construct_workload_cvs(results, qs, n_experts, n_layers, k, l, target_cvs,
                                                     max_repeats=max_repeats, verbose=verbose)
            metrics = evaluate_workload_quality_cvs(target_cvs, cvs)
            workload['obtained_cvs'] = cvs
            workload['mae'] = metrics['mae']
            workload['prompts'] = p
            workload['indices'] = indices
            workload['percent_unique_prompts'] = len(set(indices))/len(indices)
            workloads[l][a] = workload
    if verbose:
        print('done!')
    return workloads
