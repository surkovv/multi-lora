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
from multilora.system.LoRALLM import LoRALLM
from multilora.system.processors import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GlobalState:
    tasks = set()
    llm: LoRALLM = None
    processor: Processor = None

state = GlobalState()
model_loaded = asyncio.Event()

class GenerateRequestModel(BaseModel):
    prompt: str
    lora_adapter_id: str

class FineTuneRequestModel(BaseModel):
    prompts: List[str]
    adapter_name: str

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    state.llm = LoRALLM()
    state.processor = InterleaveProcessor(state.llm)
    model_loaded.set()

@app.on_event("shutdown")
async def shutdown_event():
    # Cancel batch processors
    for task in state.tasks:
        task.cancel()
    await asyncio.gather(*state.tasks, return_exceptions=True)

@app.post("/generate")
async def generate(request: GenerateRequestModel):
    logger.info(f"Received generate request for adapter {request.lora_adapter_id}")
    
    # Wait for model to be loaded
    if not model_loaded.is_set():
        logger.error("Model not loaded yet")
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    if request.lora_adapter_id not in state.llm.active_adapters:
        logger.error(f"Adapter {request.lora_adapter_id} not found")
        raise HTTPException(status_code=404, detail="LoRA adapter not found")
    
    future = asyncio.Future()
    
    state.processor.add_generate_request(
        GenerateRequest(
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
    
    state.processor.add_finetune_request(FineTuneRequest(
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
    return {"adapters": list(state.llm.active_adapters.keys())}


if __name__ == "__main__":
    logger.info("Starting batch server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 