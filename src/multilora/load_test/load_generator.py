import asyncio
import httpx
import time
import numpy as np
import random
from statistics import mean
from datasets import load_dataset
from multilora.load_test.config import (
    ENDPOINT, GENERATE_RATE, FINE_TUNE_RATE, FINE_TUNE_SIZE, TEST_DURATION,
    NUM_LORA_ADAPTERS, DATASETS
)

# Load datasets
finetome_dataset = load_dataset("mlabonne/FineTome-100k", split='train[:100]')
bitext_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split='train[:100]')
guanaco_dataset = load_dataset("mlabonne/guanaco-llama2-1k", split='train[:100]')

# Map dataset names to their actual datasets
DATASET_MAP = {
    "finetome": finetome_dataset,
    "bitext": bitext_dataset,
    "guanaco": guanaco_dataset
}

# Global latency tracking
latencies = []
fine_tune_latencies = []
active_tasks = set()


def get_prompt(dataset_name, index):
    dataset = DATASET_MAP[dataset_name]
    if dataset_name == "finetome":
        return str(dataset[index]["conversations"])
    elif dataset_name == "bitext":
        return f"Q: {dataset[index]['instruction']}\nA: {dataset[index]['response']}"
    else:  # guanaco
        return dataset[index]["text"]

async def generate_worker(client, dataset, lora_adapter_id, prompt_index):
    prompt = get_prompt(dataset, prompt_index)
    start = time.time()
    try:
        resp = await client.post(
            f"{ENDPOINT}/generate",
            json={"prompt": prompt, "lora_adapter_id": str(lora_adapter_id)},
            timeout=300.0  # Explicit timeout for this request
        )
        latency = time.time() - start
        latencies.append(latency)
        return resp
    except httpx.ReadTimeout:
        print(f"Timeout while generating with adapter {lora_adapter_id}")
        return None
    except Exception as e:
        print(f"Error in generate_worker: {e.__class__.__name__}: {str(e)}")
        return None

async def fine_tune_worker(client, dataset, adapter_id):
    prompts = [get_prompt(dataset, i) for i in range(FINE_TUNE_SIZE)]
    start = time.time()
    try:
        resp = await client.post(
            f"{ENDPOINT}/fine_tune", 
            json={"prompts": prompts, "adapter_name": adapter_id},
            timeout=300.0  # Explicit timeout for this request
        )
        latency = time.time() - start
        fine_tune_latencies.append(latency)
        return resp
    except httpx.ReadTimeout:
        print(f"Timeout while fine-tuning")
        return None
    except Exception as e:
        print(f"Error in fine_tune_worker: {e.__class__.__name__}: {str(e)}")
        return None

async def generate_loop(client, start_time):
    prompt_indices = {dataset: 0 for dataset in DATASETS}
    
    while time.time() - start_time < TEST_DURATION:
        dataset = random.choice(list(DATASET_MAP.keys()))
        adapter_id = f"adapter_{random.randint(0, NUM_LORA_ADAPTERS - 1)}" 
        task = asyncio.create_task(
            generate_worker(client, dataset, adapter_id, prompt_indices[dataset])
        )
        task.add_done_callback(active_tasks.discard)
        active_tasks.add(task)
        prompt_indices[dataset] = (prompt_indices[dataset] + 1) % 100  # Cycle through first 100 examples
        report()
        
        await asyncio.sleep(60.0 / GENERATE_RATE)

async def fine_tune_loop(client, start_time):
    adapter_cnt = 0
    next_fine_tune = start_time
    
    while time.time() - start_time < TEST_DURATION:
        if time.time() >= next_fine_tune:
            dataset = random.choice(list(DATASET_MAP.keys()))
            task = asyncio.create_task(fine_tune_worker(client, dataset, f"adapter_finetune_{adapter_cnt}"))
            task.add_done_callback(active_tasks.discard)
            active_tasks.add(task)
            next_fine_tune += 60 / FINE_TUNE_RATE
            adapter_cnt += 1
            report()

        await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting

async def load_generator():
    async with httpx.AsyncClient(timeout=300.0) as client:
        start_time = time.time()
        
        # Create tasks for each type of load
        tasks = []
        if GENERATE_RATE > 0:
            tasks.append(generate_loop(client, start_time))
        if FINE_TUNE_RATE > 0:
            tasks.append(fine_tune_loop(client, start_time))
        
        # Run all tasks concurrently
        if tasks:
            await asyncio.gather(*tasks)
        
        # Wait for remaining tasks to complete
        if active_tasks:
            await asyncio.gather(*active_tasks)

def report():
    print("\n=== Load Test Results ===")
    
    # Generate request statistics
    print(f"Total generate requests: {len(latencies)}")
    print(f"Active tasks: {len(active_tasks)}")
    if latencies:
        print(f"Mean latency: {mean(latencies):.3f}s")
        print(f"99th percentile latency: {np.percentile(latencies, 99):.3f}s")
    
    # Fine-tune statistics
    print(f"\nFine-tune requests: {len(fine_tune_latencies)}")
    if fine_tune_latencies:
        print(f"Mean fine-tune latency: {mean(fine_tune_latencies):.3f}s")
        print(f"99th percentile fine-tune latency: {np.percentile(fine_tune_latencies, 99):.3f}s")

if __name__ == "__main__":
    asyncio.run(load_generator())
    report() 