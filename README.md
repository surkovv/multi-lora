# Multi-LoRA Training and Inference System

This system provides a FastAPI-based server for training and using multiple LoRA adapters with a base language model. It supports concurrent training and inference requests, making it suitable for load testing and production use.

## Features

- Multiple LoRA adapter support
- Concurrent training and inference
- RESTful API endpoints
- Automatic adapter management
- Support for different datasets

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment:
```bash
# Set your Hugging Face token if using gated models
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

3. Start the server:
```bash
uvicorn src.multilora.server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Generate Text
```bash
POST /generate
{
    "prompt": "Your prompt here",
    "lora_adapter_id": "adapter_0"
}
```

### Fine-tune New Adapter
```bash
POST /fine_tune
{
    "prompts": ["Training prompt 1", "Training prompt 2", ...],
    "adapter_name": "optional_custom_name"
}
```

### List Available Adapters
```bash
GET /adapters
```

## Load Testing

The system includes a load testing module that can be used to test the performance of the server. To run the load test:

```bash
python -m src.multilora.load_test.load_generator
```

## Configuration

The system uses the following default configuration:
- Base model: Llama-2-7b-hf
- LoRA adapters are stored in the `lora_adapters` directory
- Default LoRA configuration:
  - r=16
  - alpha=32
  - dropout=0.05
  - Target modules: q_proj, v_proj

## Notes

- The system requires a GPU with sufficient VRAM to run the base model
- LoRA adapters are automatically loaded on server startup
- Training is done in a simple loop; for production use, consider implementing a more sophisticated training pipeline
- The system uses half-precision (float16) by default for better memory efficiency
