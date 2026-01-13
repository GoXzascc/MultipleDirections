#!/bin/bash

set -e

# Script to run linear curvature measurement for multiple models
# The linear.py script automatically processes all concepts defined in CONCEPT_CATEGORIES

models=(
  # "google/gemma-2-2b"
  # "Qwen/Qwen3-1.7B"
  # "EleutherAI/pythia-70m"
  # "EleutherAI/pythia-160m"
  "Qwen/Qwen3-0.6B"
)

# Optional: You can add command-line arguments here
# For example:
# ALPHA_MIN=1e-3
ALPHA_MAX=1e9
# ALPHA_POINTS=200

echo "Starting linear curvature measurement for ${#models[@]} models"
echo "=========================================="

for model in "${models[@]}"; do
  echo ""
  echo "Running for model: $model"
  echo "----------------------------------------"
  
  uv run src/step_length.py --model "$model" --alpha_max "$ALPHA_MAX"
  uv run src/directional_change.py --model "$model" --alpha_max "$ALPHA_MAX"
  uv run src/linear.py --model "$model" --alpha_max "$ALPHA_MAX"
  
  echo "✓ Successfully completed: $model"
done

echo ""
echo "=========================================="
echo "All models processed!"
