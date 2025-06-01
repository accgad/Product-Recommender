"""
Configuration file for the product recommender model.
"""
import os

# Model Configuration
MODEL_CONFIG = {
    # Base model
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # Training parameters
    "batch_size": 4,
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "max_length": 512,
    "weight_decay": 0.01,
    "fp16": True,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "scheduler": "cosine",
    
    # LoRA parameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    
    # Evaluation and saving
    "eval_steps": 100,
    "save_steps": 200,
    "logging_steps": 50,
    
    # Output directory
    "output_dir": "model/output"
}

# Data paths
DATA_PATHS = {
    "train": "data/processed/train.json",
    "validation": "data/processed/val.json",
    "test": "data/processed/test.json",
    "metadata": "data/processed/product_lookup.json",
    "raw_data": "data/raw/"
}

# Chat template for formatting conversations
CHAT_TEMPLATE = "<|user|>\n{user_input}\n<|assistant|>\n"

# Generation parameters for inference
GENERATION_CONFIG = {
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True
}

# API Configuration (for Flask app)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}
