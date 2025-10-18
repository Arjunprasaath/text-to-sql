import re
import json
import math
import torch
import wandb
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor



def parse_schema(schema_str: str) -> Dict[str, List[str]]:
    """Parses a schema string into a dictionary mapping table names to column lists."""
    schema = {}
    parts = [p.strip() for p in schema_str.split('|') if p.strip()]
    for part in parts:
        if ';' not in part:
            continue
        table_name_str, columns_str = [s.strip() for s in part.split(';', 1)]
        columns = [c.strip() for c in columns_str.split(',')]
        schema[table_name_str.lower()] = columns
    return schema


def normalize(name: str) -> str:
    """Normalize names: lowercase and remove underscores/spaces."""
    return re.sub(r'[\s_]+', '', name.strip().lower())


def extract_aliases(sql_query: str) -> Dict[str, str]:
    """Extract aliases (alias -> real table)."""
    alias_pattern = re.compile(
        r'\bFROM\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)'
        r'|\bJOIN\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)',
        re.IGNORECASE
    )
    aliases = {}
    for match in alias_pattern.finditer(sql_query):
        t1, a1, t2, a2 = match.groups()
        if t1 and a1:
            aliases[a1.lower()] = t1.lower()
        if t2 and a2:
            aliases[a2.lower()] = t2.lower()
    return aliases


def get_sql_template_and_reference(schema_str: str, sql_query: str) -> Tuple[str, str]:
    """Generates reference and template for given SQL and schema."""
    schema = parse_schema(schema_str)
    aliases = extract_aliases(sql_query)

    # Flatten schema info
    all_tables = list(schema.keys())
    all_columns = {c for cols in schema.values() for c in cols}

    # Normalization maps
    norm_table_map = {normalize(t): t for t in all_tables}
    norm_col_map = {normalize(c): c for c in all_columns}

    # Extract potential tokens (tables, columns, aliases)
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|"[^"]+"|\*', sql_query)

    found_tables = set()
    found_columns = set()

    for token in tokens:
        tok = token.strip('"').lower()
        tok_norm = normalize(tok)
        # Match tables
        if tok_norm in norm_table_map:
            found_tables.add(norm_table_map[tok_norm])
        # Match columns
        if tok_norm in norm_col_map:
            found_columns.add(norm_col_map[tok_norm])

    # Include aliases’ base tables if they appear
    for alias, table in aliases.items():
        if table in schema:
            found_tables.add(table)

    # --- Build Reference ---
    reference_parts = []
    seen_pairs = set()

    for table in found_tables:
        table_cols = schema.get(table, [])
        cols_for_table = [
            col for col in table_cols
            if normalize(col) in {normalize(fc) for fc in found_columns}
        ]
        if (table, tuple(cols_for_table)) not in seen_pairs:
            reference_parts.append(
                f"{table}; {', '.join(cols_for_table)}" if cols_for_table else f"{table};"
            )
            seen_pairs.add((table, tuple(cols_for_table)))

    reference = f"| {' | '.join(reference_parts)} |" if reference_parts else "| |"

    # --- Build Template ---
    template = sql_query

    # Replace alias.column → alias._
    for alias, table in aliases.items():
        if table in schema:
            for col in schema[table]:
                for variant in [col, col.replace(' ', '_')]:
                    pattern = rf'\b{alias}\.{re.escape(variant)}\b'
                    template = re.sub(pattern, f'{alias}._', template, flags=re.IGNORECASE)

    # Replace table.column → table._
    for table in found_tables:
        for col in schema.get(table, []):
            for variant in [col, col.replace(' ', '_')]:
                pattern = rf'\b{table}\.{re.escape(variant)}\b'
                template = re.sub(pattern, f'{table}._', template, flags=re.IGNORECASE)

    # Replace remaining names with '_'
    all_names = sorted(found_tables.union(found_columns), key=len, reverse=True)
    for name in all_names:
        for variant in [name, name.replace(' ', '_')]:
            template = re.sub(rf'\b{re.escape(variant)}\b', '_', template, flags=re.IGNORECASE)

    return reference, template

class SpiderDataset(Dataset):
    """
    Dataset class for inference on Spider dataset.
    Returns samples in a format suitable for model prediction.
    """
    def __init__(self, data_path, table_data_path, tokenizer, max_length = 512):
        """Load the evaluation dataset and schema information."""
        print(f"Loading dataset from {data_path}...")
        # super.__init__()
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
        
        print(f"Loaded {len(self.data)} examples for inference.")
    
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
    
    def _create_input_prompt(self, schema: str, query: str) -> str:
        """
        Create the input prompt for the model.
        Modify this function based on your model's expected input format.
        """
        prompt = f"""Given the database schema and question, plan and generate sql query. The plan consist of 2 componenets:
        1. Reference: It specifies the exact table and column names from the schema that will appear in teh query.
            Example: | table name 1; column name 1, column name 2, ... , column name n | table name 2; column name 1, ... , column name n|
        2. Template: Fully formed SQL query with slots in place of table and column names.
            Exmaple: SELECT _ FROM _ WHERE _ NOT IN (SELECT _ FROM _ )
        
        ### Question:
        Schema: {schema}
        Question: {query}
        
        ### Response:
        """        
        return prompt

    def _create_response(self, reference: str, template: str, gold_sql: str) -> str:
        """
        Create the expected response.
        """
        response = f"""<think>
        Reference: {reference}
        Template: {template}
        </think>
        SQL Query: {gold_sql}"""

        return response
    
    def __getitem__(self, idx):
        """Return query, schema, db_id for inference."""
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data)))]
        
        sample = self.data[idx]
        query = sample['question']
        db_id = sample['db_id']
        gold_sql = sample.get('query', '')
        formatted_schema = self.schema_map.get(db_id, "")
        reference, template = get_sql_template_and_reference(formatted_schema, gold_sql)
        prompt = self._create_input_prompt(formatted_schema, query)
        response = self._create_response(reference, template, gold_sql)
        full_text = prompt + response

        encodings = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        labels = encodings['input_ids'].clone()

        prompt_encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        prompt_length = int(prompt_encoding['attention_mask'].sum().item())
        # Mask prompt tokens
        labels[0, :prompt_length] = -100

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

class SlantedTriangularScheduler:
    """Slanted triangular learning rate scheduler."""
    
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps, cut_frac=0.1, ratio=32):
        """
        Args:
            optimizer: The optimizer to schedule
            base_lr: Base learning rate
            total_steps: Total training steps
            warmup_steps: Number of warmup steps
            cut_frac: Fraction of iterations to increase LR (warmup phase)
            ratio: Ratio of lowest to highest LR
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.current_step = 0
        
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Warmup phase: linear increase
            p = self.current_step / self.warmup_steps
        else:
            # Cooldown phase: slanted triangular decay
            cooldown_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = self.current_step - self.warmup_steps
            p = 1.0 - (steps_since_warmup / cooldown_steps)
        
        # Slanted triangular formula
        lr = self.base_lr * (1 + p * (self.ratio - 1)) / self.ratio
        return lr
    
    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr

def load_model_and_tokenizer(model_path):
    """
    Load your trained model and tokenizer.
    
    MODIFY THIS FUNCTION based on your model architecture.
    This is a placeholder that you should replace with your actual model loading code.
    """
    print("\nLoading model and tokenizer...")
     
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_size = "left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype = torch.bfloat16, device_map = 'cuda')
    return model, tokenizer

# def train_epoch(model, dataloader, optimizer, scheduler, device):
#     """
#     Train for one epoch.
#     """
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(dataloader, desc = "Training")

#     for batch in progress_bar:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

#         loss = outputs.loss
#         total_loss += loss.item()

#         loss.backward()
#         torch.nn.util.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.set_postfiz({'loss: ': loss.item()})
#     return total_loss / len(dataloader)

def train_step(model, batch, optimizer, scheduler, device, gradient_accumulation_steps, step):
    """
    Execute one training step.
    """
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss / gradient_accumulation_steps
    loss.backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current_lr = scheduler.step()
        optimizer.zero_grad()
    else:
        current_lr = scheduler.get_lr()
    
    return loss.item() * gradient_accumulation_steps, current_lr

def evaluate(model, dataloader, device, max_eval_batches = None):
    """
    Evaluate the model.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_eval_batches and i >= max_eval_batches:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    batch_size = 32
    gradient_accumulation_steps = 2
    base_lr = 1e-4
    total_steps = 10000
    warmup_steps = 1000
    eval_every = 500
    max_eval_batches = 50
    max_length = 512

    train_data_path = "spider_data/train_spider.json"
    dev_data_path = "spider_data/dev.json"
    table_path = "spider_data/tables.json"
    model_path = "/projects/p32722/Models/Qwen2.5-0.5B-Instruct"

    # Extract model name and create output directory with model name and batch size
    model_name = model_path.rstrip('/').split('/')[-1]
    output_dir = f"./trained_model_{model_name}_bs{batch_size}/"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="text-to-sql-training",
        name=f"training_with_plan_{model_name}_bs{batch_size}",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "base_lr": base_lr,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "max_length": max_length,
            "optimizer": "Adafactor",
            "scheduler": "SlantedTriangular",
            "training_type": "with_planning"
        }
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load dataset
    training_dataset = SpiderDataset(train_data_path, table_path, tokenizer, max_length)
    dev_dataset = SpiderDataset(dev_data_path, table_path, tokenizer, max_length)

    train_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
    dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = False)

    optimizer = Adafactor(model.parameters(), lr = base_lr, scale_parameter = False, relative_step = False, warmup_init = False)

    # Setup slanted triangular scheduler
    scheduler = SlantedTriangularScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training for {total_steps} steps")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Base LR: {base_lr}, Warmup steps: {warmup_steps}")
    print(f"Evaluation every {eval_every} steps")
    print(f"{'='*70}\n")
    
    best_eval_loss = float('inf')
    running_loss = 0
    step = 0
    
    # Create iterator that cycles through data
    train_iter = iter(train_dataloader)
    
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator when dataset is exhausted
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        loss, current_lr = train_step(
            model, batch, optimizer, scheduler, device,
            gradient_accumulation_steps, step
        )
        running_loss += loss

        step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            'loss': f'{loss:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Log training metrics to wandb
        wandb.log({
            "train/loss": loss,
            "train/learning_rate": current_lr,
            "train/step": step
        }, step=step)
        
        # Evaluate
        if step % eval_every == 0:
            avg_train_loss = running_loss / eval_every
            eval_loss = evaluate(model, dev_dataloader, device, max_eval_batches)

            print(f"\n{'='*70}")
            print(f"Step {step}/{total_steps}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"{'='*70}\n")

            # Log evaluation metrics to wandb
            wandb.log({
                "eval/loss": eval_loss,
                "eval/avg_train_loss": avg_train_loss,
                "eval/step": step
            }, step=step)

            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print(f"New best evaluation loss! Saving model...")

                save_path = output_path / f"checkpoint_step_{step}"
                save_path.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(save_path)
                print(f"Model saved to {save_path}\n")

                # Log best model info to wandb
                wandb.log({
                    "best/eval_loss": best_eval_loss,
                    "best/step": step
                }, step=step)

            running_loss = 0.0
    
    progress_bar.close()
    
    # Final save
    print("\nTraining completed! Saving final model...")
    final_path = output_path / "final_model_" + model_path.split('/')[-1]
    final_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")

    # Log final summary to wandb
    wandb.run.summary["best_eval_loss"] = best_eval_loss
    wandb.run.summary["total_steps"] = step
    wandb.run.summary["final_model_path"] = str(final_path)

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()