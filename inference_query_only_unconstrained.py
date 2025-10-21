"""
Query-only Unconstrained Inference Script
This script performs standard beam search decoding without constraints.
Used to replicate "Query-only unconstrained" results from Table 2.
Expected results (Qwen2-0.5B): EM 64.9%, EX 68.3%
"""

import re
import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpiderDataset:
    """Dataset for inference on Spider."""
    def __init__(self, data_path, table_data_path):
        print(f"Loading dataset from {data_path}...")

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        with open(table_data_path, 'r') as f:
            table_list = json.load(f)

        # Build schema map
        self.schema_map = {}
        for db_schema in table_list:
            db_id = db_schema['db_id']
            formatted_schema = self._format_schema(db_schema)
            self.schema_map[db_id] = formatted_schema

        print(f"Loaded {len(self.data)} examples for inference.")

    def _format_schema(self, db_schema):
        """Format database schema into string representation."""
        table_names = db_schema['table_names_original']
        column_names_info = db_schema['column_names_original']

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

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'query': sample['question'],
            'schema': self.schema_map.get(sample['db_id'], ""),
            'db_id': sample['db_id'],
            'gold_sql': sample.get('query', '')
        }


def create_input_prompt(question: str, schema: str) -> str:
    """Create input prompt (matches training format)."""
    prompt = f"""Given the database schema and question, generate the SQL query.

### Question:
Schema: {schema}
Question: {question}

### Response:
SQL Query: """
    return prompt


def clean_sql_query(sql_string: str) -> str:
    """Clean SQL query by removing markdown and extra whitespace."""
    # Extract SQL from code blocks
    sql_block_pattern = r'```\s*(?:sql)?\s*(.*?)```'
    match = re.search(sql_block_pattern, sql_string, flags=re.IGNORECASE | re.DOTALL)

    if match:
        cleaned = match.group(1)
    else:
        cleaned = sql_string
        cleaned = re.sub(r'```\s*sql\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)

    # Remove newlines and extra spaces
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('\r', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    return cleaned


def save_predictions(predictions, db_ids, output_path):
    """Save predictions in evaluation format."""
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        for pred_sql, db_id in zip(predictions, db_ids):
            f.write(f"{pred_sql}\t{db_id}\n")
    print(f"Saved {len(predictions)} predictions.")


def run_inference(model, tokenizer, dataset, beam_size=4, max_new_tokens=200):
    """Run unconstrained inference with beam search."""
    predictions = []
    db_ids = []
    model.eval()

    print(f"\nRunning UNCONSTRAINED inference on {len(dataset)} examples...")
    print(f"Using beam search with beam_size={beam_size}")

    with torch.inference_mode():
        for i in tqdm(range(len(dataset)), desc="Inference"):
            sample = dataset[i]
            question = sample['query']
            schema = sample['schema']
            db_id = sample['db_id']

            input_text = create_input_prompt(question, schema)
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            # Standard beam search decoding (unconstrained)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                early_stopping=True,
                do_sample=False
            )

            # Decode and extract SQL
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract SQL from response
            if "SQL Query:" in generated_text:
                predicted_sql = generated_text.split("SQL Query:")[-1].strip()
            else:
                predicted_sql = generated_text[len(input_text):].strip()

            predicted_sql = clean_sql_query(predicted_sql)
            predictions.append(predicted_sql)
            db_ids.append(db_id)

    return predictions, db_ids


def main():
    # Configuration
    data_path = "spider_data/test.json"  # Use test set for final evaluation
    table_path = "spider_data/tables.json"
    model_path = "/projects/p32722/Models/text2sql/trained_model_query_only_Qwen2.5-7B-Instruct_bs1/final_model_query_only_Qwen2.5-7B-Instruct"  # Best checkpoint
    output_dir = "./predictions/"

    beam_size = 4  # Standard beam search
    max_new_tokens = 2048

    # Extract model name and create output file name with model name
    model_name = model_path.rstrip('/').split('/')[-1]
    output_file = f"query_only_unconstrained_predictions_{model_name}.txt"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = SpiderDataset(data_path, table_path)

    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )

    # Run inference
    predictions, db_ids = run_inference(model, tokenizer, dataset, beam_size, max_new_tokens)

    # Save predictions
    save_path = output_path / output_file
    save_predictions(predictions, db_ids, save_path)

    print(f"\n{'='*60}")
    print("Query-only UNCONSTRAINED Inference Complete!")
    print(f"Predictions saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
