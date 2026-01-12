#!/bin/bash

# Script to run linear curvature measurement for multiple models
# The linear.py script automatically processes all concepts defined in CONCEPT_CATEGORIES

models=(
  "google/gemma-2-2b"
  "Qwen/Qwen3-1.7B"
#   "EleutherAI/pythia-70m"
  "EleutherAI/pythia-160m"
)

# Optional: You can add command-line arguments here
# For example:
# ALPHA_MIN=1e-3
# ALPHA_MAX=1e7
# ALPHA_POINTS=200

echo "Starting linear curvature measurement for ${#models[@]} models"
echo "=========================================="

for model in "${models[@]}"; do
  echo ""
  echo "Running for model: $model"
  echo "----------------------------------------"
  
  uv run src/linear.py \
    --model "$model"
  
  if [ $? -eq 0 ]; then
    echo "✓ Successfully completed: $model"
  else
    echo "✗ Failed: $model"
    # Uncomment the next line if you want to stop on first error
    # exit 1
  fi
done

echo ""
echo "=========================================="
echo "All models processed!"
