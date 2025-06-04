import torch
from multilora.lora_layer import MultiLoRALayerSTK
from multilora.lora_model import LoRAModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

N_adapters = 32

def create_lora_het_stk(in_features, out_features, adapter_ids):
    return MultiLoRALayerSTK(in_features, out_features, adapter_ids, ranks=[32] * N_adapters)

class LoRALLM:

    MODEL_NAME = "openai-community/gpt2-medium"
    active_adatpers = list(map(str, range(N_adapters)))
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.lora_model = LoRAModel(
            self.base_model, 
            target_modules=["c_attn", "c_proj"], 
            lora_factory=create_lora_het_stk).cuda().to(torch.bfloat16)
        self.lora_model.freeze_base_model()

    def generate(self, batch: List[str], adapters: List[int]):
        adapters = torch.tensor(adapters, device='cuda')
        batch = self.tokenizer(batch, return_tensors='pt').to('cuda')
        result = self.lora_model.generate(**batch, adapter_ids=adapters)
        return self.tokenizer.batch_decode(result, skip_special_tokens=True)

    def finetune(self, batch: List[str], adapters: List[int]):
        pass