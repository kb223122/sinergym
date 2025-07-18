#!/usr/bin/env python3
"""
Simple example script to run the perfect evaluation without command line arguments.
Modify the paths and settings below as needed.
"""

import sys
import os

# ============== CONFIGURATION - MODIFY THESE ==============

# Model paths and settings
MODEL_DIR = '/Users/z5543337/Desktop/work/PPO-DistilBERT-trainable-20250709-025523-res1'
MODEL_NAME = 'ppo_distilbert_20250709-025523.zip'
BERT_MODE = 'trainable'  # Options: 'fixed', 'trainable', 'partial'
NUM_EPISODES = 12
OUTPUT_DIR = 'evaluation_results'

# ============== RUN EVALUATION ==============

def run_evaluation():
    """Run the evaluation with the configured settings."""
    
    # Set up command line arguments for the main script
    sys.argv = [
        'perfect_evaluation_script.py',
        '--model_dir', MODEL_DIR,
        '--model_name', MODEL_NAME,
        '--bert_mode', BERT_MODE,
        '--num_episodes', str(NUM_EPISODES),
        '--output_dir', OUTPUT_DIR
    ]
    
    # Import and run the main evaluation script
    from perfect_evaluation_script import main
    main()

if __name__ == "__main__":
    print("🚀 Running PPO DistilBERT Evaluation")
    print("=" * 50)
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"BERT Mode: {BERT_MODE}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 50)
    
    # Check if model file exists
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        print("Please check the MODEL_DIR and MODEL_NAME variables above.")
        exit(1)
    
    # Run evaluation
    try:
        run_evaluation()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise