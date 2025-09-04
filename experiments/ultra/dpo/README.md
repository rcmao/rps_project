# Ultra DPO Comparison Script

This script compares Ultra DPO baseline best responses against RPS best responses using GPT-based judging.

## Setup

1. **Set OpenAI API Key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Install required packages:**
```bash
pip install pandas tqdm openai
```

## Usage

```bash
cd /mnt/rps_project/experiments/ultra/dpa

python3 compare_ultra_dpo_rps_vs_baseline.py \
    --baseline_dir /mnt/rps_project/data/ultra/dpo_baseline_best_response \
    --rps_dir /mnt/rps_project/data/ultra/dpo_rps_best_resposne \
    --output_dir /mnt/rps_project/data/ultra/dpo_ultra_compare_result \
    --model gpt-4o-mini \
    --seed 42
```

## Script Features

### Adapted for Ultra Dataset
- **File matching**: Handles ultra dataset naming conventions
  - Baseline files: `dpo_responses_v{X}_merged.csv`
  - RPS files: `v{X}_best_response.csv`
- **Column normalization**: Automatically maps different column names
  - `response` → `best_response`
  - `v1_p`/`main_v1` → `v1`
  - `v2_p`/`main_v2` → `v2`

### Processing Pipeline
1. **Data Loading**: Load and merge baseline/RPS data by `prompt_id`
2. **Pairwise Generation**: Create randomized A/B comparisons
3. **GPT Judging**: Use Appendix A.2 prompt for preference evaluation
4. **Results Analysis**: Calculate win rates and generate summary

### Output Files
- `v{X}_pairwise.jsonl`: Input data for GPT judging
- `v{X}_judged.jsonl`: GPT judgment results
- `ultra_comparison_summary.csv`: Aggregated win rate statistics

## Command Line Options

- `--baseline_dir`: Directory containing baseline CSV files
- `--rps_dir`: Directory containing RPS CSV files  
- `--output_dir`: Directory to write results
- `--model`: OpenAI model for judging (default: gpt-4o-mini)
- `--seed`: Random seed for reproducibility (default: 42)
- `--directions`: Specific directions to process (default: v3-v10)
- `--max_samples`: Maximum number of samples to process per direction (for testing/resuming)
- `--max_tokens`: Maximum tokens for GPT response (default: 256, original HelpSteer used 32)

## Batch Processing and Resume Feature

The script supports **batch processing** and **automatic resuming**:

### Test with Small Batches
```bash
# Process only 100 samples per direction for testing
python3 compare_ultra_dpo_rps_vs_baseline.py \
    --baseline_dir /mnt/rps_project/data/ultra/dpo_baseline_best_response \
    --rps_dir /mnt/rps_project/data/ultra/dpo_rps_best_resposne \
    --output_dir /mnt/rps_project/data/ultra/dpo_ultra_compare_result \
    --model gpt-4o-mini \
    --max_samples 100
```

### Resume Processing
```bash
# Continue processing remaining samples (automatic resume)
python3 compare_ultra_dpo_rps_vs_baseline.py \
    --baseline_dir /mnt/rps_project/data/ultra/dpo_baseline_best_response \
    --rps_dir /mnt/rps_project/data/ultra/dpo_rps_best_resposne \
    --output_dir /mnt/rps_project/data/ultra/dpo_ultra_compare_result \
    --model gpt-4o-mini \
    --max_samples 500
```

### Process All Remaining
```bash
# Process all remaining samples (no max_samples limit)
python3 compare_ultra_dpo_rps_vs_baseline.py \
    --baseline_dir /mnt/rps_project/data/ultra/dpo_baseline_best_response \
    --rps_dir /mnt/rps_project/data/ultra/dpo_rps_best_resposne \
    --output_dir /mnt/rps_project/data/ultra/dpo_ultra_compare_result \
    --model gpt-4o-mini
```

### Resume Logic
- Script automatically detects existing `*_judged.jsonl` files
- Skips already processed samples based on their `id`
- Continues from where it left off
- Saves progress every 50 judgments for safety

## Key Improvements vs Original HelpSteer Script

### Reduced TIE Rate
- **Improved parsing**: Multiple strategies to extract A/B/Tie judgments
- **Increased max_tokens**: Default 256 tokens vs original 32 tokens
- **Better error handling**: More robust judgment extraction

### Parsing Strategies (in order)
1. Look for "More aligned: A/B/Tie" pattern
2. Find standalone A/B/Tie at line ends  
3. Search for any A/B/Tie in response
4. Check last word as fallback
5. Debug output for failed parsing

### Usage with Custom Token Limit
```bash
# Use original HelpSteer settings (32 tokens)
python3 compare_ultra_dpo_rps_vs_baseline.py ... --max_tokens 32

# Use improved settings (256 tokens, recommended)
python3 compare_ultra_dpo_rps_vs_baseline.py ... --max_tokens 256

# Use even more tokens for complex responses
python3 compare_ultra_dpo_rps_vs_baseline.py ... --max_tokens 512
```

## Example Output

```
📂 Baseline files: 8 | RPS files: 8
🎯 Processing directions: ['v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']

============================================================
🎯 Processing v3
============================================================
Found baseline files: ['dpo_responses_v3_merged.csv']
Found RPS files: ['v3_best_response.csv']
✅ Loaded 512 rows from dpo_responses_v3_merged.csv
✅ Loaded 512 rows from v3_best_response.csv
📊 Baseline DF: 512 rows, RPS DF: 512 rows
📊 Merged 512 matching prompt_ids for v3
✅ Generated 512 pairs: /path/to/v3_pairwise.jsonl
📊 Raw A/B/TIE: 245/231/36
   Baseline wins: 245 | RPS wins: 231 | Ties: 36 | RPS win-rate (excl. ties): 0.485

🏆 Overall Results:
   Total comparisons: 4096
   Baseline wins: 2048
   RPS wins: 1920
   Ties: 128
   RPS win-rate (excl. ties): 0.484
```
