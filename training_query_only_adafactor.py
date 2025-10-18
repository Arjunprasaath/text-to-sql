"""
Query-only Training Script for Text-to-SQL with AdaFactor
This script trains models WITHOUT planning (no references, no templates).
Uses AdaFactor optimizer and Slanted Triangular Learning Rate scheduler.
"""

import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor


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
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data)))]

        sample = self.data[idx]
        question = sample['question']
        db_id = sample['db_id']
        gold_sql = sample.get('query', '')
        formatted_schema = self.schema_map.get(db_id, "")

        prompt = self._create_input_prompt(formatted_schema, question)
        response = self._create_response(gold_sql)
        full_text = prompt + response

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


class SlantedTriangularScheduler:
    """
    Slanted Triangular Learning Rate Scheduler
    Increases learning rate linearly for cut_frac of training,
    then decreases linearly for the rest.
    """
    def __init__(self, optimizer, num_training_steps, cut_frac=0.1, ratio=32, last_epoch=-1):
        """
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total number of training steps
            cut_frac: Fraction of training for increasing LR (default: 0.1 = 10%)
            ratio: Ratio of lowest to highest LR (default: 32)
            last_epoch: The index of last epoch
        """
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.cut = int(num_training_steps * cut_frac)
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.step_count < self.cut:
            # Increasing phase
            p = self.step_count / self.cut
        else:
            # Decreasing phase
            p = 1 - ((self.step_count - self.cut) / (self.num_training_steps - self.cut))

        lr_factor = (1 + p * (self.ratio - 1)) / self.ratio
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def step(self):
        """Update learning rate."""
        self.step_count += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


def load_model_and_tokenizer(model_path):
    """Load model and tokenizer."""
    print(f"\nLoading model and tokenizer from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    return model, tokenizer


def train_step(model, batch, optimizer, scheduler, device, gradient_accumulation_steps, step):
    """Execute one training step."""
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
        scheduler.step()
        optimizer.zero_grad()

    current_lr = optimizer.param_groups[0]['lr']
    return loss.item() * gradient_accumulation_steps, current_lr


def evaluate(model, dataloader, device, max_eval_batches=None):
    """Evaluate the model."""
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
    # Configuration with AdaFactor and Slanted Triangular scheduler
    batch_size = 32
    gradient_accumulation_steps = 16
    base_lr = 1e-3  # AdaFactor typically uses higher LR than AdamW
    total_steps = 10000
    cut_frac = 0.2  # 20% of training for warmup (2000 steps)
    ratio = 32  # Ratio for slanted triangular
    eval_every = 500
    max_eval_batches = 50
    max_length = 512

    # Early stopping parameters to prevent overfitting
    patience = 5  # Number of evaluations without improvement before stopping
    min_delta = 0.001  # Minimum change to qualify as an improvement

    # Paths (UPDATE THESE)
    train_data_path = "spider_data/train_spider.json"
    dev_data_path = "spider_data/dev.json"
    table_path = "spider_data/tables.json"
    model_path = "/projects/p32722/Models/Qwen2.5-0.5B-Instruct"

    # Extract model name and create output directory with model name and batch size
    model_name = model_path.rstrip('/').split('/')[-1]
    output_dir = f"/projects/p32722/Models/text2sql/trained_model_query_only_adafactor_{model_name}_bs{batch_size}/"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load dataset
    training_dataset = QueryOnlySpiderDataset(train_data_path, table_path, tokenizer, max_length)
    dev_dataset = QueryOnlySpiderDataset(dev_data_path, table_path, tokenizer, max_length)

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer: AdaFactor
    # AdaFactor is a memory-efficient optimizer that adapts learning rates
    optimizer = Adafactor(
        model.parameters(),
        lr=base_lr,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,  # Use fixed learning rate with our scheduler
        scale_parameter=False,
        warmup_init=False
    )

    # Setup slanted triangular scheduler
    scheduler = SlantedTriangularScheduler(
        optimizer,
        num_training_steps=total_steps,
        cut_frac=cut_frac,
        ratio=ratio
    )

    # Training loop
    print(f"\n{'='*70}")
    print(f"QUERY-ONLY TRAINING WITH ADAFACTOR (No Plans)")
    print(f"Using AdaFactor optimizer with Slanted Triangular scheduler")
    print(f"Starting training for {total_steps} steps")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Base LR: {base_lr}")
    print(f"Cut fraction: {cut_frac} (warmup for {int(total_steps * cut_frac)} steps)")
    print(f"LR ratio: {ratio}")
    print(f"Evaluation every {eval_every} steps")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
    print(f"{'='*70}\n")

    best_eval_loss = float('inf')
    running_loss = 0
    step = 0
    patience_counter = 0
    eval_history = []

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

        # Evaluate
        if step % eval_every == 0:
            avg_train_loss = running_loss / eval_every
            eval_loss = evaluate(model, dev_dataloader, device, max_eval_batches)
            eval_history.append(eval_loss)

            print(f"\n{'='*70}")
            print(f"Step {step}/{total_steps}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")

            # Check for improvement
            if eval_loss < best_eval_loss - min_delta:
                improvement = best_eval_loss - eval_loss
                best_eval_loss = eval_loss
                patience_counter = 0
                print(f"Improvement: {improvement:.4f} - Saving model...")

                save_path = output_path / f"checkpoint_step_{step}"
                save_path.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement (patience: {patience_counter}/{patience})")

                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at step {step}")
                    print(f"Best evaluation loss: {best_eval_loss:.4f}")
                    print(f"{'='*70}\n")
                    break

            print(f"{'='*70}\n")
            running_loss = 0.0

    progress_bar.close()

    # Final save
    print("\nTraining completed! Saving final model...")
    final_path = output_path / "final_model_query_only_adafactor"
    final_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")

    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'eval_history': eval_history,
            'best_eval_loss': best_eval_loss,
            'total_steps': step,
            'config': {
                'optimizer': 'AdaFactor',
                'scheduler': 'SlantedTriangular',
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'base_lr': base_lr,
                'cut_frac': cut_frac,
                'ratio': ratio
            }
        }, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
