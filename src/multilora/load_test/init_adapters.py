import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "openai-community/gpt2"
LORA_BASE_DIR = Path("lora_adapters")
NUM_ADAPTERS = 3  # Match NUM_LORA_ADAPTERS from config.py

def create_adapter(model, adapter_id: str, adapter_path: Path):
    """Create and save a LoRA adapter with random initialization."""
    logger.info(f"Creating adapter {adapter_id}")
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Create PEFT model with unique adapter name
    peft_model = get_peft_model(model, peft_config, adapter_name=adapter_id)
    
    # Save adapter
    peft_model.save_pretrained(str(adapter_path), selected_adapters=[adapter_id])

    logger.info(f"Saved adapter {adapter_id} to {adapter_path}")

def main():
    # Create base directory if it doesn't exist
    LORA_BASE_DIR.mkdir(exist_ok=True)
    
    # Load base model
    logger.info(f"Loading model {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logger.info("Model loaded successfully")
    
    # Create adapters
    for i in range(NUM_ADAPTERS):
        adapter_id = f"adapter_{i}"
        adapter_path = LORA_BASE_DIR
        create_adapter(model, adapter_id, adapter_path)
    
    logger.info(f"Created {NUM_ADAPTERS} adapters in {LORA_BASE_DIR}")

if __name__ == "__main__":
    main() 