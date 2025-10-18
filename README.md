# Text-to-SQL with Constrained Decoding

This repository implements text-to-SQL models based on the paper "Improving Text-to-SQL with Constrained Decoding of Satisfiable Plans and Faithful Queries" (TACL 2025).

## Overview

The codebase supports training and evaluation of text-to-SQL models on the Spider dataset using Qwen language models. It implements both **Plan-based** and **Query-only** approaches.

---

## File Structure

```
text-to-sql/
|_ README.md                                    # This file
|_ training.py                                  # Plan + Query training (with references & templates)
|_ inference.py                                 # Basic inference script
|_ training_query_only.py                       # Query-only training (no plans)
|_ inference_query_only_unconstrained.py        # Query-only unconstrained inference
|_ spider_data/                                 # Spider dataset (to be downloaded)
   |_ train_spider.json
   |_ dev.json
   |_ test.json
   |_ tables.json
|_ trained_model/                               # Plan + Query models
|_ trained_model_query_only/                    # Query-only models
```

---

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers accelerate
```

### 2. Download Spider Dataset

Download from the [official Spider repository](https://yale-lily.github.io/spider) and organize:

```bash
mkdir spider_data
# Place train_spider.json, dev.json, test.json, tables.json in spider_data/
```

### 3. Update Model Paths

In all scripts, update the `model_path` variable to point to your Qwen model:

```python
model_path = "/path/to/your/Qwen2.5-0.5B-Instruct"  # Or 1.5B, 7B variants
```

---

## Training

### Option 1: Plan + Query Training (Original)

**File**: `training.py`

Trains models to generate **plans** (references + templates) before SQL queries.

```bash
python training.py
```

**Key Features**:
- Generates references (table/column names) and templates (SQL structure)
- Uses custom parser to extract plans from gold SQL
- Implements slanted triangular learning rate schedule
- Saves best checkpoint based on dev loss

**Configuration** (in `training.py`):
- Batch size: 32 (effective 64 with gradient accumulation)
- Learning rate: 1e-4
- Total steps: 10,000
- Warmup: 1,000 steps
- Optimizer: AdaFactor

**Output**: `./trained_model/checkpoint_step_{step}/`

---

### Option 2: Query-only Training (for Table 2 replication)

**File**: `training_query_only.py`

Trains models to directly generate SQL queries **without plans**.

```bash
python training_query_only.py
```

**Key Differences from Plan + Query**:
- No reference or template generation
- Simpler prompt: just question + schema � SQL
- Same hyperparameters as paper specification

**Output**: `./trained_model_query_only/checkpoint_step_{step}/`

---

## Inference

### Option 1: Basic Inference

**File**: `inference.py`

Simple inference script for any trained model.

```bash
python inference.py
```

**Features**:
- Loads model and generates SQL predictions
- Cleans output (removes markdown, newlines)
- Saves predictions to `./predictions/`

---

### Option 2: Query-only Unconstrained Inference

**File**: `inference_query_only_unconstrained.py`

Replicates **Query-only unconstrained** results from Table 2.

```bash
python inference_query_only_unconstrained.py
```

**Features**:
- Standard beam search (beam_size=4)
- No constraints on generation
- Expected results (Qwen2-0.5B): **EM 64.9%, EX 68.3%**

**Configuration**:
```python
data_path = "spider_data/test.json"                          # Spider test set
model_path = "./trained_model_query_only/checkpoint_step_5000"
beam_size = 4
max_new_tokens = 200
```

**Output**: `./predictions/query_only_unconstrained_predictions.txt`

Format: `{predicted_sql}\t{db_id}` (one per line)

---

## Training Details

### Hyperparameters (from Paper)

Following Appendix 8.1 of the paper:

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Gradient accumulation | 2 steps |
| Effective batch size | 64 |
| Learning rate | 1e-4 |
| Total training steps | 10,000 |
| Warmup steps | 1,000 |
| Optimizer | AdaFactor |
| LR schedule | Slanted Triangular |
| Max sequence length | 512 |
| Evaluation frequency | Every 500 steps |

### Learning Rate Schedule

**Slanted Triangular Scheduler**:
- **Warmup phase** (0-1,000 steps): Linear increase
- **Cooldown phase** (1,000-10,000 steps): Slanted triangular decay

Formula:
```
if step < warmup_steps:
    p = step / warmup_steps
else:
    p = 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)

lr = base_lr * (1 + p * (ratio - 1)) / ratio
```

---

## Data Formatting

### Plan + Query Format (training.py)

**Input**:
```
Given the database schema and question, plan and generate sql query. The plan consist of 2 components:
1. Reference: It specifies the exact table and column names from the schema that will appear in the query.
   Example: | table name 1; column name 1, column name 2 | table name 2; column name 1 |
2. Template: Fully formed SQL query with slots in place of table and column names.
   Example: SELECT _ FROM _ WHERE _ NOT IN (SELECT _ FROM _ )

### Question:
Schema: {schema}
Question: {question}

### Response:
```

**Output**:
```
<think>
Reference: | representative; representative_id, name | election; representative_id |
Template: SELECT _ FROM _ WHERE _ NOT IN (SELECT _ FROM _)
</think>
SQL Query: SELECT name FROM representative WHERE representative_id NOT IN (SELECT representative_id FROM election)
```

---

### Query-only Format (training_query_only.py)

**Input**:
```
Given the database schema and question, generate the SQL query.

### Question:
Schema: {schema}
Question: {question}

### Response:
```

**Output**:
```
SQL Query: SELECT name FROM representative WHERE representative_id NOT IN (SELECT representative_id FROM election)
```

---

## Evaluation

After generating predictions, evaluate using the official Spider evaluation script:

```bash
# Download evaluation script
git clone https://github.com/taoyds/spider.git
cd spider

# Run evaluation
python evaluation.py \
    --gold ../spider_data/test_gold.sql \
    --pred ../predictions/query_only_unconstrained_predictions.txt \
    --db ../spider_data/database/ \
    --table ../spider_data/tables.json

python  spider/evaluation.py --gold dev_gold_test.sql --pred predictions/dev_prediction_Qwen2.5-7B-Instruct.txt --etype all --db spider_data/database --table spider_data/tables.json 

# Test Eval
python spider/evaluation.py   --gold spider_data/test_gold.sql   --pred predictions/query_only_unconstrained_predictions.txt   --etype all   --db spider_data/test_database   --table spider_data/test_tables.json 
```

**Metrics**:
- **Exact Match (EM)**: String match with gold SQL
- **Execution Accuracy (EX)**: Result match when executed on database

---

## Key Implementation Details

### 1. Schema Formatting

Schemas are serialized as:
```
| table1 ; *, col1, col2, col3 | table2 ; *, col4, col5 |
```

### 2. Reference & Template Extraction (Plan-based)

**Reference Extraction**:
- Parse SQL to find all table and column names
- Resolve aliases (e.g., `t1` � `table1`)
- Group columns by their parent tables

**Template Generation**:
- Replace `table.column` � `table._`
- Replace `alias.column` � `alias._`
- Replace remaining schema elements with `_`

### 3. Training Strategy

Both training scripts:
1. Load Qwen model in bfloat16
2. Use AdaFactor optimizer (no weight decay)
3. Train on prompt+response, masking prompt tokens in loss
4. Evaluate every 500 steps
5. Save best checkpoint based on dev loss
6. Cycle through training data until reaching 10,000 steps

---

## Expected Results Replication

### For Qwen2-0.5B on Spider Test Set:

| Experiment | Script | Expected EM | Expected EX |
|------------|--------|-------------|-------------|
| Query-only unconstrained | `inference_query_only_unconstrained.py` | 64.9% | 68.3% |
| Query-only QCD | *To be implemented* | 66.9% | 70.6% |
| Plan + Query PLANIT | *Requires full implementation* | 70.9% | 75.3% |

### Training Steps:

1. **Train Query-only model**:
   ```bash
   python training_query_only.py
   ```
   - Monitor dev loss and select best checkpoint (typically around step 5000)

2. **Run unconstrained inference**:
   ```bash
   python inference_query_only_unconstrained.py
   ```
   - Update model path to best checkpoint

3. **Evaluate**:
   ```bash
   python spider/evaluation.py --gold ... --pred ...
   ```

---

## Notes

- **GPU Required**: Training requires CUDA-capable GPU (tested on H100, A100)
- **Memory**: ~20GB GPU memory for 0.5B model, ~40GB for 1.5B model
- **Training Time**: ~3-4 hours on A100 for 10,000 steps
- **Beam Search**: Default beam size is 4 (as per paper)

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{planit2025,
  title={Improving Text-to-SQL with Constrained Decoding of Satisfiable Plans and Faithful Queries},
  journal={Transactions of the Association for Computational Linguistics (TACL)},
  year={2025}
}
```

---

## Future Work

To fully replicate all Table 2 results:

1.  Query-only unconstrained (implemented)
2.  Query-only QCD (requires PICARD-style constraint implementation)
3.  Plan + Query PQCD (requires plan constraint implementation)
4.  Plan + Query PLANIT (requires satisfiability & faithfulness constraints)

See the paper (Sections 2.1-2.4) for detailed algorithm descriptions.
