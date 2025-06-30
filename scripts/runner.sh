#!/bin/bash

# ============================================
# Configuration Section
# Define the shared variables once here
# to avoid repetition and ensure consistency
# ============================================

DATASET_NAME=20NG        # Change this to your dataset name
NUM_TOPICS=50            # Set the number of topics to use
NUM_TOPWORDS=15          # Number of top words to evaluate per topic
CONFIG_FILE=config.yaml  # Path to the main config file
CLIENT=OpenAI            # LLM evaluation client (e.g., OpenAI, Gemini)

# ============================================
# Run Topic Modeling
# ============================================

echo "Running topic modeling on dataset: $DATASET_NAME with $NUM_TOPICS topics..."

python main.py \
  --dataset_name $DATASET_NAME \
  --num_topics $NUM_TOPICS \
  --config $CONFIG_FILE

# ============================================
# Run LLM-based Topic Evaluation
# ============================================

echo "Running LLM evaluation on the generated topics..."

python evaluations/llm_evaluation/llm_eval.py \
  -r tm_datasets \
  --dataset_name $DATASET_NAME \
  --num_topics $NUM_TOPICS \
  --num_topwords $NUM_TOPWORDS \
  --client $CLIENT

echo "✅ All tasks completed successfully."
