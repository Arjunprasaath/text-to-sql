"""
Query-only Training Script using Transformers Trainer API
This script trains models WITHOUT planning (no references, no templates).
Uses the high-level Trainer API for cleaner, more maintainable code.
"""

import json
import torch
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


class QueryOnlySpiderDataset(Dataset):
    """
    Dataset for Query-only training (no plans).
    Directly maps: question + schema -> SQL query
    """
    def __init__(self, data_path, table_data_path, tokenizer, max_length=512):
        print(f"Loading Query-only dataset from {data_path}...")

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        with open(table_data_path, 'r') as f:
            table_list = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build schema map
        self.schema_map = {}
        for db_schema in table_list:
            db_id = db_schema['db_id']
            formatted_schema = self._format_schema(db_schema)
            self.schema_map[db_id] = formatted_schema

        print(f"Loaded {len(self.data)} examples for Query-only training.")

    def _format_schema(self, db_schema):
        """Format database schema into string representation."""
        table_names = db_schema['table_names']
        column_names_info = db_schema['column_names']

        table_to_columns = {name: [] for name in table_names}
        for col_info in column_names_info:
            table_idx = col_info[0]
            col_name = col_info[1]

            if table_idx >= 0:
                table_name = table_names[table_idx]
                table_to_columns[table_name].append(col_name)

        schema_parts = []
        for table_name in table_names:
            columns = table_to_columns[table_name]
            full_columns = ['*'] + columns
            columns_str = ", ".join(full_columns)
            schema_parts.append(f"| {table_name} ; {columns_str} ")

        return "".join(schema_parts) + "|"

    def __len__(self):
        return len(self.data)

    def _create_input_prompt(self, schema: str, question: str) -> str:
        """Create input prompt without planning (Query-only)."""
        prompt = f"""Given the database schema and question, generate the SQL query.

### Question:
Schema: {schema}
Question: {question}

### Response:
"""
        return prompt

    def _create_response(self, gold_sql: str) -> str:
        """Create response with just the SQL query (no plans)."""
        response = f"SQL Query: {gold_sql}"
        return response

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        db_id = sample['db_id']
        gold_sql = sample.get('query', '')
        formatted_schema = self.schema_map.get(db_id, "")

        prompt = self._create_input_prompt(formatted_schema, question)
        response = self._create_response(gold_sql)
        full_text = prompt + response

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        labels = encodings['input_ids'].clone()

        # Mask prompt tokens (only train on response)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        prompt_length = int(prompt_encoding['attention_mask'].sum().item())
        labels[0, :prompt_length] = -100

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def main():
    # Configuration
    train_data_path = "spider_data/train_spider.json"
    dev_data_path = "spider_data/dev.json"
    table_path = "spider_data/tables.json"
    model_path = "/projects/p32722/Models/Qwen2.5-0.5B-Instruct"
    output_dir = "/projects/p32722/Models/text2sql/trained_model_query_only_trainer/"

    # Hyperparameters
    max_length = 512
    batch_size = 32
    gradient_accumulation_steps = 4  # Effective batch size = 64
    learning_rate = 5e-5
    weight_decay = 0.01
    max_steps = 10000
    warmup_steps = 1000
    eval_steps = 500
    save_steps = 500
    logging_steps = 50

    print(f"\n{'='*70}")
    print(f"QUERY-ONLY TRAINING using Transformers Trainer API")
    print(f"{'='*70}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = QueryOnlySpiderDataset(train_data_path, table_path, tokenizer, max_length)
    eval_dataset = QueryOnlySpiderDataset(dev_data_path, table_path, tokenizer, max_length)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto'  # Automatically distribute across available GPUs
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Training hyperparameters
        num_train_epochs=1,  # We use max_steps instead
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # Optimizer settings
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  # Cosine scheduler with warmup

        # Optimizer
        optim="adamw_torch",  # Use PyTorch's AdamW
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,  # Gradient clipping

        # Evaluation and logging
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",

        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Performance
        bf16=True,  # Use bfloat16 for training
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Misc
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        seed=42,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Print training info
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Model: {model_path.split('/')[-1]}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max steps: {max_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR scheduler: cosine")
    print(f"  Optimizer: AdamW")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Save every: {save_steps} steps")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}\n")

    # Train
    print("Starting training...\n")
    trainer.train()

    # Save final model
    print("\n\nTraining completed! Saving final model...")
    final_path = Path(output_dir) / "final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

    # Print best checkpoint info
    print(f"\n{'='*70}")
    print(f"Training Summary:")
    print(f"  Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"  Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
