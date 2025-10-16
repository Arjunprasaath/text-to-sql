import re
import json
import torch
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpiderDataset:
    """
    Dataset class for inference on Spider dataset.
    Returns samples in a format suitable for model prediction.
    """
    def __init__(self, data_path, table_data_path):
        """Load the evaluation dataset and schema information."""
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
    
    def __getitem__(self, idx):
        """Return query, schema, db_id for inference."""
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self.data)))]
        
        sample = self.data[idx]
        query = sample['question']
        db_id = sample['db_id']
        formatted_schema = self.schema_map.get(db_id, "")
        
        return {
            'query': query,
            'schema': formatted_schema,
            'db_id': db_id,
            'gold_sql': sample.get('query', '')  # For gold file generation
        }


def create_input_prompt(query: str, schema: str) -> str:
    """
    Create the input prompt for the model.
    Modify this function based on your model's expected input format.
    """
    prompt = f"""Given the database schema and question, generate ONLY the SQL query.
    Schema: {schema}
    Question: {query}
    SQL Query:"""
    return prompt


def save_predictions(predictions: List[str], output_path: str):
    """Save predictions in the format expected by evaluation.py (one SQL per line)."""
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        for pred_sql in predictions:
            f.write(pred_sql + '\n')
    print(f"Saved {len(predictions)} predictions.")


def load_model_and_tokenizer(model_path):
    """
    Load your trained model and tokenizer.
    
    MODIFY THIS FUNCTION based on your model architecture.
    This is a placeholder that you should replace with your actual model loading code.
    """
    print("\nLoading model and tokenizer...")
     
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_size = "left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype = torch.bfloat16, device_map = 'cuda')
    return model, tokenizer

def clean_sql_query(sql_string: str) -> str:
    """
    Clean SQL query by extracting only the SQL code from markdown blocks
    or removing markdown markers, newlines, and extra whitespace.
    
    Args:
        sql_string: Raw SQL string that may contain ```sql, ```, \n, and other text
    
    Returns:
        Cleaned SQL query as a single line with normalized spacing
    """
    # First, try to extract SQL from within code blocks
    # Pattern to match content between ```sql and ``` or ``` and ```
    sql_block_pattern = r'```\s*(?:sql)?\s*(.*?)```'
    match = re.search(sql_block_pattern, sql_string, flags=re.IGNORECASE | re.DOTALL)
    
    if match:
        # Extract only the SQL content from the code block
        cleaned = match.group(1)
    else:
        # No code block found, clean the entire string
        cleaned = sql_string
        # Remove any stray ``` markers
        cleaned = re.sub(r'```\s*sql\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'```', '', cleaned)
    
    # Remove all newline characters (\n, \r\n, \r)
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('\r', ' ')
    
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

def run_inference(model, tokenizer, dataset, max_output_length, temperature, batch_size = 4) -> List[Tuple[str, str]]:
    """
    Run inference on the dataset using the provided model.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        dataset: SpiderInferenceDataset instance
        args: Command line arguments
    
    Returns:
        List of tuples: [(predicted_sql, db_id), ...]
    """    
    predictions = []
    model.eval()
    print(f"\nRunning inference on {len(dataset)} examples...")

    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset) - 1000), desc="Inference"):
            sample = dataset[i]
            query = sample['query']
            schema = sample['schema']
            input_text = create_input_prompt(query, schema)

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_length=max_output_length,
                temperature=temperature,
                early_stopping=True,
            )

            predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "SQL Query:" in predicted_sql:
                predicted_sql = predicted_sql.split("SQL Query:")[-1].strip()
            predictions.append(predicted_sql)
    return predictions

def main():
    temperature = 0.1
    max_output_length = 200
    data_path = "spider_data/dev.json"
    table_path = "spider_data/tables.json"
    output_dir = "./predictions/"
    pred_file = data_path.split('/')[-1][:-5] + "_prediction" # dev_prediction.txt
    
    model_path = "/projects/p32722/Models/Qwen2.5-0.5B-Instruct"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    # Load dataset
    dataset = SpiderDataset(data_path, table_path)
    
    # Create gold file
    # create_gold_file(dataset, gold_file)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Run inference
    predictions = run_inference(model, tokenizer, dataset, max_output_length, temperature)
    # print(predictions)

    # Save predictions
    save_predictions(predictions, output_dir + pred_file + '_' + model_path.split('/')[-1] + ".txt")
    
    print(f"\n{'='*60}")
    print("Benchmarking Complete!")
    print(f"Predictions saved to: {pred_file}.txt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()