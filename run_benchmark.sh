#!/bin/bash

# Configuration
DATA_DIR="data"
RESULTS_DIR="results"
SCRIPTS_DIR="scripts"

# List of models to evaluate
# Note: Use your specific LiteLLM identifiers here.
# For local models, ensuring your VLLM/Ollama server is running.
SOTA_MODELS=(
    #"gemini/gemini-3"
    # "openai/gpt-5.2"
    #"anthropic/claude-4.5-sonnet"
)

CONSUMER_MODELS=(
    "huggingface/Qwen/Qwen3.5-9B:together"
)

ARABIC_MODELS=(
)

ALL_MODELS=("${SOTA_MODELS[@]}" "${CONSUMER_MODELS[@]}" "${ARABIC_MODELS[@]}")

# ALL_MODELS=(
# )

# Check for API Keys
echo "--- Environment Check ---"
if [ -z "$GEMINI_API_KEY" ] || [ -z "$OPENAI_API_KEY" ] || [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "[WARN] One or more API keys are missing. Ensure GEMINI_API_KEY, OPENAI_API_KEY, and ANTHROPIC_API_KEY are set."
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo -e "\n--- Starting Benchmark Execution ---\n"

for MODEL in "${ALL_MODELS[@]}"; do
    echo "========================================================="
    echo "PROCESSING MODEL: $MODEL"
    echo "========================================================="
    
    # 1. Run Inference
    python3 "$SCRIPTS_DIR/evaluate.py" \
        --model "$MODEL" \
        --data_dir "$DATA_DIR" \
        --results_dir "$RESULTS_DIR"
    
    # 2. Generate Stats for this model
    # The evaluate.py script saves results in: results/<model_name_safe>/
    SAFE_MODEL_NAME=$(echo "$MODEL" | sed 's/\//_/g')
    MODEL_RESULTS_PATH="$RESULTS_DIR/$SAFE_MODEL_NAME"
    
    # if [ -d "$MODEL_RESULTS_PATH" ]; then
    #     python3 "$SCRIPTS_DIR/generate_stats.py" \
    #         --results_dir "$MODEL_RESULTS_PATH"
    # else
    #     echo "[ERROR] Results directory not found for $MODEL at $MODEL_RESULTS_PATH"
    # fi
    
    echo -e "Finished $MODEL\n"
done

echo "========================================================="
echo "ALL MODELS PROCESSED"
echo "Check the '$RESULTS_DIR' folder for JSON results and CSV stats."
echo "========================================================="
