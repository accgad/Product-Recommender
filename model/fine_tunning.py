"""
Script for fine-tuning TinyLlama on product recommendation data
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import MODEL_CONFIG, DATA_PATHS, CHAT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama for product recommendations")
    parser.add_argument("--train_file", type=str, default="data/processed/train.json",
                        help="Path to training data file")
    parser.add_argument("--val_file", type=str, default="data/processed/val.json",
                        help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, default="model/output",
                        help="Output directory for the model")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    return parser.parse_args()


def load_data(data_path):
    """Load and format data for training"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        if "messages" in item and len(item["messages"]) >= 2:
            messages = item["messages"]
            if messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
                formatted_data.append({
                    "user_input": messages[0]["content"],
                    "assistant_response": messages[1]["content"]
                })
    
    print(f"Loaded {len(formatted_data)} conversation pairs from {data_path}")
    return Dataset.from_list(formatted_data)


def format_chat(examples, tokenizer, chat_template):
    """Format conversations for training"""
    conversations = []
    
    for user_input, assistant_response in zip(examples["user_input"], examples["assistant_response"]):
        # Create the full conversation
        if chat_template:
            user_formatted = chat_template.format(user_input=user_input)
            full_conversation = user_formatted + assistant_response
        else:
            # Fallback format if no template
            full_conversation = f"<|user|>\n{user_input}\n<|assistant|>\n{assistant_response}"
        
        conversations.append(full_conversation)
    
    # Tokenize conversations
    tokenized = tokenizer(
        conversations,
        truncation=True,
        max_length=512,  # Reasonable max length for TinyLlama
        padding="max_length",
        return_tensors="pt",
    )
    
    # Set labels = input_ids for causal LM fine-tuning
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Convert tensors to lists for Dataset compatibility
    return {k: v.tolist() for k, v in tokenized.items()}


def prepare_datasets(tokenizer, train_path, val_path, chat_template):
    """Prepare train and validation datasets"""
    print(f"Loading training data from {train_path}")
    train_dataset = load_data(train_path)
    
    print(f"Loading validation data from {val_path}")
    val_dataset = load_data(val_path)
    
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    
    # Tokenize datasets
    print("Tokenizing training data...")
    train_tokenized = train_dataset.map(
        lambda examples: format_chat(examples, tokenizer, chat_template),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    print("Tokenizing validation data...")
    val_tokenized = val_dataset.map(
        lambda examples: format_chat(examples, tokenizer, chat_template),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    return train_tokenized, val_tokenized


def train(args):
    """Fine-tune TinyLlama with PEFT/LoRA"""
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load chat template from config, fallback if not available
    try:
        chat_template = CHAT_TEMPLATE
    except (NameError, ImportError):
        print("Warning: CHAT_TEMPLATE not found in config, using default format")
        chat_template = "<|user|>\n{user_input}\n<|assistant|>\n"
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(tokenizer, args.train_file, args.val_file, chat_template)
    
    # Load model
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for PEFT fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,  # LoRA dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        group_by_length=True,
        report_to="none",
        save_total_limit=3,
        remove_unused_columns=False,
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return final_model_path


if __name__ == "__main__":
    args = parse_args()
    train(args)
