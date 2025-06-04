# Default configuration for the load test submodule

# Endpoint configuration
ENDPOINT = "http://localhost:8000"

# Load test parameters
GENERATE_RATE = 0  # generates per minute
FINE_TUNE_RATE = 30  # fine-tunes per minute
FINE_TUNE_SIZE = 32 # prompts per fine-tune
TEST_DURATION = 300  # seconds

# Dataset and adapter configuration
NUM_LORA_ADAPTERS = 3  # Number of pretrained LoRA adapters (indexed 0 to N-1)
DATASETS = [
    "finetome",    # FineTome-100k dataset
    "bitext",      # Bitext customer support dataset
    "guanaco"      # Guanaco dataset
] 