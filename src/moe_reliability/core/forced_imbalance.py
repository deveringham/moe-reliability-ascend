###
# forced_imbalance.py
#
# Construction of MoE checkpoints with forced router imbalance: a bias on
# expert 0 in every router so that routing collapses onto a single expert,
# which in turn induces a straggler rank under expert parallelism.
# Dylan Everingham
# 10.08.2026
###

from transformers import AutoConfig
import torch
from .hf_models import load_model


def imbalance_pretrained_moe(model_id, imbalance_level, save_path):

    model, tokenizer = load_model(model_id, max_memory=None, enable_bnb=False)
    config = AutoConfig.from_pretrained(model_id)
    
    # If no imbalance, we're done.
    if imbalance_level > 0:
        with torch.no_grad():
            
            # Apply bias to embeddings such that sum is positive
            model.model.embed_tokens.weight.data += 1.0
    
            for name, module in model.named_modules():
                if "gate" in name:
                    
                    # Zero-center expert weights
                    row_means = module.weight.data.mean(dim=1, keepdim=True,
                                                        dtype=module.weight.data.dtype)
                    module.weight.data -= row_means
                    
                    # Apply imbalance
                    module.weight[0, :] += imbalance_level
                
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
