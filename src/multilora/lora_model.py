import torch
import torch.nn as nn
import transformers
import types
import os
from peft import get_peft_model, LoraConfig, PeftModel

from multilora import LoRALayer, MultiLoRALayerMaskingHom, MultiLoRALayerMaskingHomEfficient

class ModuleSum(nn.Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
    
    def forward(self, *args, **kwargs):
        return self.module1(*args, **kwargs) + self.module2(*args, **kwargs)


class LoRAModel(nn.Module):
    def __init__(self, base_model, target_modules, lora_factory):
        """
        Wraps a given LLM model and injects LoRA layers into specified layers.

        Args:
            base_model (nn.Module): The original LLM model.
            lora_config (dict): Dictionary with LoRA configuration.
        """
        super().__init__()
        self.base_model = base_model
        self.target_modules = target_modules
        self.adapter_ids = torch.tensor([], dtype=torch.long).to('cuda')
        self.lora_factory = lora_factory

        self._inject_lora_layers()

    def _get_n_features_by_module(self, module):
        if isinstance(module, nn.Linear):
            return module.in_features, module.out_features
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            return (
                module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
            )
        else:
            raise NotImplementedError(f"Module of type {type(module)} are not supported")

    def _inject_lora_layers(self):
        """
        Injects LoRA layers into the specified layers of the model.
        """
        target_modules = self.target_modules

        modules_to_replace = {}

        for name, module in self.base_model.named_modules():
            if any(layer_name in name for layer_name in target_modules):
                in_features, out_features = self._get_n_features_by_module(module)
                # Wrap the original linear layer
                wrapped_layer = ModuleSum(
                    module,  # Original layer
                    self.lora_factory(in_features, out_features, self.adapter_ids),
                )
                modules_to_replace[name] = wrapped_layer
            
        for name, new_module in modules_to_replace.items():
            parent_module, child_name = self._get_parent_module(name)
            setattr(parent_module, child_name, new_module)

    def _get_parent_module(self, module_name):
        """
        Given a module name, returns its parent module and the attribute name.
        """
        module_parts = module_name.split(".")
        parent = self.base_model
        for part in module_parts[:-1]:  # Navigate to the parent module
            parent = getattr(parent, part)
        
        return parent, module_parts[-1]  # Return parent and attribute name

    def freeze_base_model(self):
        """
        Freezes all parameters in the base model except for LoRA layers.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALayer):
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, *args, adapter_ids, **kwargs):
        self.adapter_ids.data = adapter_ids
        return self.base_model(*args, **kwargs)
    
    def generate(self, *args, adapter_ids=None, **kwargs):
        if adapter_ids is None:
            raise ValueError("`adapter_ids` must be provided for generation.")

        self.adapter_ids.data = adapter_ids

        original_forward = self.base_model.forward

        def patched_forward(model_self, *f_args, **f_kwargs):
            self.adapter_ids.data = adapter_ids
            return original_forward(*f_args, **f_kwargs)

        self.base_model.forward = types.MethodType(patched_forward, self.base_model)

        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self.base_model.forward = original_forward
