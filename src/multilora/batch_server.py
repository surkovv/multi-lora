from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os
import json
from pathlib import Path
import asyncio
import threading
import uvicorn
import logging
from collections import defaultdict
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
MODEL_NAME = "openai-community/gpt2-medium"
LORA_BASE_DIR = Path("lora_adapters")
LORA_BASE_DIR.mkdir(exist_ok=True)

# Batch processing configuration
BATCH_WINDOW = 0.1  # seconds to wait for batching
MAX_BATCH_SIZE = 32  # maximum batch size for generation
MAX_FINE_TUNE_BATCH_SIZE = 32  # maximum batch size for fine-tuning

# Global state
model = None
tokenizer = None
active_adapters = {}
model_lock = threading.Lock()
model_loaded = threading.Event()
model_loading_executor = ThreadPoolExecutor(max_workers=1)

# Request batching queues
generate_queue = asyncio.Queue()
fine_tune_queue = asyncio.Queue()
batch_tasks = set()

@dataclass
class GenerateRequest:
    prompt: str
    adapter_id: str
    future: asyncio.Future

@dataclass
class FineTuneRequest:
    prompts: List[str]
    adapter_name: Optional[str]
    future: asyncio.Future

class GenerateRequestModel(BaseModel):
    prompt: str
    lora_adapter_id: str

class FineTuneRequestModel(BaseModel):
    prompts: List[str]
    adapter_name: Optional[str] = None

def load_model():
    global model, tokenizer
    try:
        logger.info(f"Loading model {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("Model loaded successfully")
        logger.info("Loading tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        
        # Load existing adapters
        for adapter_dir in LORA_BASE_DIR.glob("*"):
            if adapter_dir.is_dir():
                adapter_id = adapter_dir.name
                try:
                    logger.info(f"Loading adapter {adapter_id}")
                    model.load_adapter(str(adapter_dir), adapter_name=adapter_id)
                    active_adapters[adapter_id] = str(adapter_dir)
                    logger.info(f"Adapter {adapter_id} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load adapter {adapter_id}: {e}")
        model_loaded.set()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

async def process_generate_batch(batch: List[GenerateRequest]):
    if not batch:
        return
    
    # Group requests by adapter_id
    adapter_batches = defaultdict(list)
    for req in batch:
        adapter_batches[req.adapter_id].append(req)
    
    # Process each adapter's batch
    for adapter_id, requests in adapter_batches.items():
        try:
            with model_lock:
                model.set_adapter(adapter_id)
                
                # Prepare inputs
                prompts = [req.prompt for req in requests]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    max_length=64,
                    truncation=True
                ).to(model.device)
                
                # Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Decode and set results
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for req, response in zip(requests, responses):
                    req.future.set_result(response)
                    
        except Exception as e:
            logger.error(f"Error processing generate batch for adapter {adapter_id}: {e}")
            for req in requests:
                req.future.set_exception(e)

async def process_fine_tune_batch(batch: List[FineTuneRequest]):
    if not batch:
        return
    
    for req in batch:
        try:
            adapter_id = req.adapter_name or f"adapter_{len(active_adapters)}"
            adapter_path = LORA_BASE_DIR
            
            # Train adapter in thread pool
            await asyncio.get_event_loop().run_in_executor(
                model_loading_executor,
                train_adapter,
                adapter_id,
                adapter_path,
                req.prompts
            )
            req.future.set_result({"adapter_id": adapter_id, "status": "ready"})
            
        except Exception as e:
            logger.error(f"Error processing fine-tune request: {e}")
            req.future.set_exception(e)

async def generate_batch_processor():
    while True:
        batch = []
        try:
            # Get first request
            batch.append(await generate_queue.get())
            
            # Try to get more requests within the batch window
            start_time = time.time()
            while len(batch) < MAX_BATCH_SIZE and time.time() - start_time < BATCH_WINDOW:
                try:
                    req = await asyncio.wait_for(generate_queue.get(), BATCH_WINDOW)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            
            # Process the batch
            await process_generate_batch(batch)
            
        except Exception as e:
            logger.error(f"Error in generate batch processor: {e}")
            for req in batch:
                req.future.set_exception(e)

async def fine_tune_batch_processor():
    while True:
        batch = []
        try:
            # Get first request
            batch.append(await fine_tune_queue.get())
            
            # Try to get more requests within the batch window
            start_time = time.time()
            while len(batch) < MAX_FINE_TUNE_BATCH_SIZE and time.time() - start_time < BATCH_WINDOW:
                try:
                    req = await asyncio.wait_for(fine_tune_queue.get(), BATCH_WINDOW)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            
            # Process the batch
            await process_fine_tune_batch(batch)
            
        except Exception as e:
            logger.error(f"Error in fine-tune batch processor: {e}")
            for req in batch:
                req.future.set_exception(e)

def train_adapter(adapter_id: str, adapter_path: Path, training_texts: List[str]):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    with model_lock:
        try:
            active_adapters[adapter_id] = str(adapter_path)
            # Configure LoRA
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Create PEFT model
            peft_model = get_peft_model(model, peft_config, adapter_name=adapter_id)
            peft_model.train()
            
            # Training loop
            optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
            
            logger.info(f"Training adapter {adapter_id}")
            for text in training_texts:
                inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True).to(peft_model.device)
                outputs = peft_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            logger.info(f"Trained adapter {adapter_id}")
            # Save adapter
            peft_model.save_pretrained(str(adapter_path), selected_adapters=[adapter_id])
            
            # Load the trained adapter into the main model
            model.load_adapter(str(adapter_path), adapter_name=adapter_id)
        except Exception as e:
            active_adapters.pop(adapter_id, None)
            logger.error(f"Error during training adapter {adapter_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    # Start batch processors
    batch_tasks.add(asyncio.create_task(generate_batch_processor()))
    batch_tasks.add(asyncio.create_task(fine_tune_batch_processor()))
    
    # Load model in a separate thread
    model_loading_executor.submit(load_model)

@app.on_event("shutdown")
async def shutdown_event():
    # Cancel batch processors
    for task in batch_tasks:
        task.cancel()
    await asyncio.gather(*batch_tasks, return_exceptions=True)

@app.post("/generate")
async def generate(request: GenerateRequestModel):
    logger.info(f"Received generate request for adapter {request.lora_adapter_id}")
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        logger.error("Model not loaded yet")
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    if request.lora_adapter_id not in active_adapters:
        logger.error(f"Adapter {request.lora_adapter_id} not found")
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    
    # Create future for this request
    future = asyncio.Future()
    
    # Add request to batch queue
    await generate_queue.put(GenerateRequest(
        prompt=request.prompt,
        adapter_id=request.lora_adapter_id,
        future=future
    ))
    
    try:
        # Wait for the result
        response = await future
        return {"response": response}
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine_tune")
async def fine_tune(request: FineTuneRequestModel):
    logger.info(f"Received fine-tune request for {len(request.prompts)} prompts")
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        logger.error("Model not loaded yet")
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    # Create future for this request
    future = asyncio.Future()
    
    # Add request to batch queue
    await fine_tune_queue.put(FineTuneRequest(
        prompts=request.prompts,
        adapter_name=request.adapter_name,
        future=future
    ))
    
    try:
        # Wait for the result
        result = await future
        return result
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/adapters")
async def list_adapters():
    logger.info("Received request to list adapters")
    return {"adapters": list(active_adapters.keys())}

@app.get("/adapter_status/{adapter_id}")
async def get_adapter_status(adapter_id: str):
    logger.info(f"Received status request for adapter {adapter_id}")
    if adapter_id not in active_adapters:
        logger.error(f"Adapter {adapter_id} not found")
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    
    return {
        "adapter_id": adapter_id,
        "status": "ready",
        "path": active_adapters[adapter_id]
    }

if __name__ == "__main__":
    logger.info("Starting batch server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 