
import json

config = json.load(open('config.json', 'r'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from multilora import LoRAModel, MultiLoRALayerMaskingHom, MultiLoRALayerMaskingHomEfficient, MultiLoRALayerMasking, MultiLoRALayerSTK

from transformers import AutoTokenizer, AutoModelForCausalLM
from multilora.benchmarking import MultiAdapterDataset, get_bitext_dataset, get_finetome_dataset, get_guanaco_dataset, get_acp_dataset
N = 1000
model_id = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = MultiAdapterDataset([get_acp_dataset(N, tokenizer)], tokenizer)

n_datasets = 1

def create_lora_het_factory(n_adapters):
    assert n_adapters % 4 == 0, "N adapters should be divisible by 4"
    def create_lora_het(in_features, out_features, adapter_ids):
        n = n_adapters // 4
        ranks = [16] * n + [32] * n + [64] * n + [128] * n
        return MultiLoRALayerMasking(in_features, out_features, adapter_ids, ranks=ranks)
    return create_lora_het

def create_lora_het_stk_factory(n_adapters):
    assert n_adapters % 4 == 0, "N adapters should be divisible by 4"
    def create_lora_het_stk(in_features, out_features, adapter_ids):
        n = n_adapters // 4
        ranks = [16] * n + [32] * n + [64] * n + [128] * n
        return MultiLoRALayerSTK(in_features, out_features, adapter_ids, ranks=ranks)
    return create_lora_het_stk


n_adapters = config['n_adapters']
factory = create_lora_het_stk_factory if config['use_stk'] else create_lora_het_factory

model = GPT2LMHeadModel.from_pretrained(model_id, device_map="auto").to(torch.bfloat16)
lora_model = LoRAModel(model, target_modules=["c_attn", "c_proj"], lora_factory=factory(n_adapters)).cuda().to(torch.bfloat16)
lora_model.freeze_base_model()

from torch.optim import AdamW
from transformers import get_scheduler

dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=dataset.collate_fn)
device = 'cuda'

def inference_step(data):
    ids, masks, labels, adapter_ids = data
    adapter_ids = adapter_ids + torch.randint_like(adapter_ids, low=0, high=n_adapters // n_datasets - 1) * n_datasets
    adapter_ids %= n_adapters
    result = lora_model.generate(input_ids=ids.to(device), adapter_ids=adapter_ids.to(device), max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    return result.shape[0]

from tqdm import tqdm
from time import time
alpha = 0.95
start = time()

iters = 0

requests_done = 0
num_epochs = 10
for epoch in range(num_epochs):
    if iters >= 100:
        break
    for i, batch in tqdm(enumerate(dataloader)):
        requests_done += inference_step(batch)
        
        if iters % 3 == 2:
            print("requests/sec:", (requests_done) / (time() - start))
            print("iters:", iters)
        iters += 1
        if iters >= 100:
            break