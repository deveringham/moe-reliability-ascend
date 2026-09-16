###
# moe_hooks.py
#
# Classes for hooking router behavior in pretrained models.
# Dylan Everingham
# 20.03.2026
###

# Dependencies
import re
import torch

# MoE hook: attaches an arbitrary hook to router modules
class MoEHook:
    
    def __init__(self, model, n_experts=64, k=2):
        self.model = model
        self.n_experts = n_experts
        self.k = k
        self.n_routers = 0
        
        # Holds router names, hooks, outputs to be injected.
        # Indexed by router module object
        # router_outputs of size [n_layers, n_experts] and default to 0
        self.routers = {}
        
        # Find router modules
        self.register(model)
        
        # Get names of all model routers, sorted by layer
        all_names = [r["name"] for r in self.routers.values()]
        self.router_names_sorted = sorted(all_names, key=self._get_router_sorted_id_by_name)
        
        print(f"MoEHook: Model has {self.n_routers} routers each with {self.n_experts} experts and selects k={self.k} at each layer.")
    
    # Helper function to get router module for a particular layer id,
    # that is, the position of its name in router_names_sorted
    def _get_router_module(self, router_id):
        name = self.router_names_sorted[router_id]
        module = None
        for m in self.routers.keys():
            if self.routers[m]["name"] == name:
                module = m
                break
        if module == None:
            raise ValueError(f"MoEHook: No layer found in model with name {name}.")
        return module
        
    # Default hook function: does nothing
    def hook_fn(self, module, inputs, outputs):
        pass
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register(self, model):
        print(f"MoEHook: Scanning model for routers...")
        for name, module in model.named_modules():
            if self.attach_fn(name, module):
                handle = module.register_forward_hook(self.hook_fn)
                self.routers[module] = {
                    "name": name,
                    "hook": handle,
                }
        self.n_routers = len(self.routers)
        print(f"MoEHook: Attached probes to {self.n_routers} router layers.")
    
    def get_n_experts(self):
        return self.n_experts
    def get_n_routers(self):
        return self.n_routers
    def get_k(self):
        return self.k
    

# Implentation for Qwen MoE models ex. Qwen1.5-MoE-A2.7B-Chat
class MoEHookQwen(MoEHook):
    
    # Function used for identifying router modules
    def attach_fn(self, name, module):
        
        # In Qwen1.5-MoE, the router is a linear layer named 'gate'
        return name.endswith(".gate")
    
    # Get router module names, ordered by layer
    # Simply order by layer id
    def _get_router_sorted_id_by_name(self, name):
        match = re.search(r'\d+', name)
        if match == None:
            return 0
        router_id = int(match.group())
        return router_id
    

# Implentation for DeepSeek MoE models ex. DeepSeek-V2-Lite-Chat
class MoEHookDeepSeek(MoEHook):
    
    # Function used for identifying router modules
    def attach_fn(self, name, module):
        
        # In DeepSeek, as with Qwen, just look for "gate"
        return name.endswith(".gate")
    
    # Get router module names, ordered by layer
    # Simply order by layer id
    def _get_router_sorted_id_by_name(self, name):
        match = re.search(r'\d+', name)
        if match == None:
            return 0
        router_id = int(match.group())
        return router_id

    
# Implentation for Mistral MoE models ex. Mixtral-8x7B-Instruct-v0.1
class MoEHookMistral(MoEHook):
    
    # Function used for identifying router modules
    def attach_fn(self, name, module):
        
        # As with Qwen and DeepSeek, just look for "gate"
        return name.endswith(".gate")
    
    # Get router module names, ordered by layer
    # Simply order by layer id
    def _get_router_sorted_id_by_name(self, name):
        match = re.search(r'\d+', name)
        if match == None:
            return 0
        router_id = int(match.group())
        return router_id
