from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
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
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
MODEL_NAME = "openai-community/gpt2"  # Base model to use
LORA_BASE_DIR = Path("lora_adapters")
LORA_BASE_DIR.mkdir(exist_ok=True)

# Global state
model = None
tokenizer = None
active_adapters = {}
model_lock = threading.Lock()
model_loaded = threading.Event()
model_loading_executor = ThreadPoolExecutor(max_workers=1)

class GenerateRequest(BaseModel):
    prompt: str
    lora_adapter_id: str

class FineTuneRequest(BaseModel):
    prompts: List[str]
    adapter_name: str

def load_model():
    global model, tokenizer
    try:
        logger.info(f"Loading model {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        ).cuda()
        logger.info("Model loaded successfully")
        
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        
        # Load existing adapters
        for adapter_dir in LORA_BASE_DIR.glob("*"):
            if adapter_dir.is_dir():
                adapter_id = adapter_dir.name
                try:
                    logger.info(f"Loading adapter {adapter_id}")
                    # Load adapter with a unique name
                    model.load_adapter(str(adapter_dir), adapter_name=adapter_id)
                    active_adapters[adapter_id] = str(adapter_dir)
                    logger.info(f"Adapter {adapter_id} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load adapter {adapter_id}: {e}")
        model_loaded.set()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    # Load model in a separate thread
    model_loading_executor.submit(load_model)

def report_vram():
    logger.info(f"Memory allocated: {torch.cuda.max_memory_allocated(0) // 1024 // 1024} MB")

def generate_with_adapter(prompt: str, adapter_id: str) -> str:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    with model_lock:
        try:
            # Switch to the requested adapter
            model.set_adapter(adapter_id)
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True).to(model.device)
            
            logger.info(f"Generating with adapter {adapter_id}")
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            logger.info(f"Generated with adapter {adapter_id}")
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error during generation with adapter {adapter_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            report_vram()

def train_adapter(adapter_id: str, adapter_path: Path, training_texts: List[str]):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    with model_lock:
        try:
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
            
        except Exception as e:
            active_adapters.pop(adapter_id, None)
            logger.error(f"Error during training adapter {adapter_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            report_vram()

@app.post("/generate")
async def generate(request: GenerateRequest):
    logger.info(f"Received generate request for adapter {request.lora_adapter_id}")
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        logger.error("Model not loaded yet")
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    if request.lora_adapter_id not in active_adapters:
        logger.error(f"Adapter {request.lora_adapter_id} not found")
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    
    # Run generation in thread pool to not block the event loop
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            model_loading_executor,
            generate_with_adapter,
            request.prompt,
            request.lora_adapter_id
        )
        logger.info(f"Successfully generated response for adapter {request.lora_adapter_id}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine_tune")
async def fine_tune(request: FineTuneRequest):
    logger.info(f"Received fine-tune request for {len(request.prompts)} prompts")
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        logger.error("Model not loaded yet")
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    # Generate unique adapter ID if not provided
    adapter_id = request.adapter_name
    adapter_path = LORA_BASE_DIR
    
    # Train adapter in thread pool
    try:
        await asyncio.get_event_loop().run_in_executor(
            model_loading_executor,
            train_adapter,
            adapter_id,
            adapter_path,
            request.prompts
        )
        logger.info(f"Successfully trained adapter {adapter_id}")
        return {"adapter_id": adapter_id, "status": "ready"}
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
    logger.info("Starting server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")