#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra DPO: RPS vs Baseline comparison

This script compares Ultra DPO baseline best responses against RPS best responses, 
using the Appendix A.2 judge prompt from the paper (preference-aligned A/B/Tie).

Specifically adapted for the ultra dataset file naming conventions:
- Baseline files: dpo_responses_v{X}_merged.csv
- RPS files: v{X}_best_response.csv

Pipeline per direction (v3..v10):
1) Load baseline and RPS CSVs and inner-join by prompt_id
2) Randomize A/B assignment and write pairwise JSONL
3) Judge pairs with GPT using weights (v1, v2) from the rows
4) Analyze win/tie rates and write summary CSV

Usage:
  python compare_ultra_dpo_rps_vs_baseline.py \
      --baseline_dir /mnt/rps_project/data/ultra/dpo_baseline_best_response \
      --rps_dir /mnt/rps_project/data/ultra/dpo_rps_best_resposne \
      --output_dir /mnt/rps_project/data/ultra/dpo_ultra_compare_result \
      --model gpt-4o-mini

Environment:
  OPENAI_API_KEY must be set. Optionally OPENAI_API_BASE.
"""

import os
import re
import json
import time
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def _find_direction_csv_ultra(files: List[Path], direction: str) -> List[Path]:
    """
    Find CSV files for a specific direction in ultra dataset.
    
    Handles patterns like:
    - dpo_responses_v3_merged.csv (baseline)
    - v3_best_response.csv (rps)
    """
    matched = []
    for p in files:
        # Check if direction appears in filename
        if direction in p.stem or direction in p.name:
            # More specific check to avoid false matches (e.g., v3 in v30)
            if re.search(rf'(?:^|_){re.escape(direction)}(?:_|$)', p.stem):
                matched.append(p)
    return matched


def _load_best_df_ultra(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load CSV and normalize column names for ultra dataset.
    
    Expected columns:
    - prompt_id, prompt, best_response/response
    - v1, v2 (weights) or v1_p, v2_p or main_v1, main_v2
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize response column name to 'best_response'
        if 'best_response' not in df.columns:
            if 'response' in df.columns:
                df = df.rename(columns={'response': 'best_response'})
        
        # Normalize weight columns to v1, v2
        if 'v1' not in df.columns:
            if 'v1_p' in df.columns:
                df = df.rename(columns={'v1_p': 'v1'})
            elif 'main_v1' in df.columns:
                df = df.rename(columns={'main_v1': 'v1'})
        
        if 'v2' not in df.columns:
            if 'v2_p' in df.columns:
                df = df.rename(columns={'v2_p': 'v2'})
            elif 'main_v2' in df.columns:
                df = df.rename(columns={'main_v2': 'v2'})
        
        # Ensure required columns exist
        needed = {'prompt_id', 'prompt', 'best_response'}
        if not needed.issubset(df.columns):
            print(f"Missing required columns in {csv_path}")
            print(f"Available columns: {list(df.columns)}")
            print(f"Required columns: {needed}")
            return None
        
        print(f"✅ Loaded {len(df)} rows from {csv_path.name}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None


def generate_pairwise_jsonl_for_gpt_judging(
    baseline_df: pd.DataFrame,
    rps_df: pd.DataFrame,
    direction_name: str,
    out_jsonl: Path,
    seed: int = 42,
) -> int:
    set_seed(seed)

    # Merge on prompt_id and prompt
    left_cols = ['prompt_id', 'prompt', 'best_response']
    right_cols = ['prompt_id', 'prompt', 'best_response']
    
    # Add weight columns if available
    if 'v1' in baseline_df.columns and 'v2' in baseline_df.columns:
        left_cols += ['v1', 'v2']
    if 'v1' in rps_df.columns and 'v2' in rps_df.columns:
        right_cols += ['v1', 'v2']

    merged = pd.merge(
        baseline_df[left_cols],
        rps_df[right_cols],
        on=['prompt_id', 'prompt'],
        suffixes=('_baseline', '_rps')
    )

    print(f"📊 Merged {len(merged)} matching prompt_ids for {direction_name}")
    
    if len(merged) == 0:
        return 0

    pairwise: List[Dict[str, Any]] = []
    for i, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Pairwise {direction_name}"):
        prompt: str = row['prompt']
        
        # Get weights, with fallback to default values
        v1 = 0.7071  # default
        v2 = 0.7071  # default
        
        if 'v1_baseline' in row:
            v1 = float(row['v1_baseline'])
        elif 'v1_rps' in row:
            v1 = float(row['v1_rps'])
        elif 'v1' in row:
            v1 = float(row['v1'])
            
        if 'v2_baseline' in row:
            v2 = float(row['v2_baseline'])
        elif 'v2_rps' in row:
            v2 = float(row['v2_rps'])
        elif 'v2' in row:
            v2 = float(row['v2'])

        # Randomly assign A/B positions
        if random.random() < 0.5:
            response_a = row['best_response_baseline']
            response_b = row['best_response_rps']
            a_is_baseline = True
        else:
            response_a = row['best_response_rps']
            response_b = row['best_response_baseline']
            a_is_baseline = False

        item = {
            'id': f'{direction_name}_{i}',
            'prompt_id': row['prompt_id'],
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'a_is_baseline': a_is_baseline,
            'direction': direction_name,
            'v1': v1,
            'v2': v2,
        }
        pairwise.append(item)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open('w', encoding='utf-8') as f:
        for it in pairwise:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    return len(pairwise)


def build_appendix_a2_prompt(prompt: str, response_a: str, response_b: str, v1: float, v2: float) -> str:
    return (
        "You are a careful evaluation judge. Decide which response better aligns with the user "
        "preference that balances two criteria with given weights:\n"
        f"- Helpfulness (weight = {v1})\n"
        f"- Verbosity (weight = {v2})\n"
        "(measured by informative detail and explanation quality; do not reward length itself. "
        "Penalize fluff, repetition, and low information density. Do not penalize concision if sufficiently informative.)\n\n"
        "Safety/correctness override:\n"
        "- Factually wrong, unsafe, or misleading content should lose regardless of verbosity.\n\n"
        "Scoring guideline (internal):\n"
        "- For each response, assess Helpfulness and Verbosity quality (0--10 integer scale), then compute S = (wh)*Helpfulness + (wv)*Verbosity. "
        "Choose the higher S. If the margin is very small, you may output \"Tie\".\n\n"
        "Output format (exactly):\n"
        "Comparison: <one-sentence comparison and reason>\n"
        "More aligned: A | B | Tie\n\n"
        f"Query: {prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}"
    )


def run_gpt_judging(input_jsonl: Path, output_jsonl: Path, model: str = "gpt-4o-mini", sleep_time: float = 0.7, max_retries: int = 3, max_samples: Optional[int] = None, max_tokens: int = 256) -> Tuple[int, int, int]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    # Set up OpenAI client with new API
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    with input_jsonl.open('r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Check if we should resume from existing results
    existing_results = []
    if output_jsonl.exists():
        print(f"📂 Found existing results file: {output_jsonl}")
        with output_jsonl.open('r', encoding='utf-8') as f:
            existing_results = [json.loads(line) for line in f]
        print(f"🔄 Resuming from {len(existing_results)} existing judgments")
    
    # Filter out already processed items
    processed_ids = {r['id'] for r in existing_results}
    remaining_data = [item for item in data if item['id'] not in processed_ids]
    
    # Apply max_samples limit to remaining data
    if max_samples is not None and max_samples > 0:
        remaining_data = remaining_data[:max_samples]
        print(f"📊 Processing {len(remaining_data)} samples (max_samples={max_samples})")
    
    if not remaining_data:
        print("✅ All samples already processed!")
        # Return counts from existing results
        a_wins = sum(1 for r in existing_results if r['judgment'] == 'A')
        b_wins = sum(1 for r in existing_results if r['judgment'] == 'B')
        ties = sum(1 for r in existing_results if r['judgment'] == 'TIE')
        return a_wins, b_wins, ties

    results: List[Dict[str, Any]] = existing_results.copy()

    for i, item in tqdm(list(enumerate(remaining_data)), total=len(remaining_data), desc=f"Judging {model}"):
        prompt = build_appendix_a2_prompt(
            prompt=item['prompt'],
            response_a=item['response_a'],
            response_b=item['response_b'],
            v1=float(item.get('v1', 0.7071)),
            v2=float(item.get('v2', 0.7071)),
        )
        judgment: str = 'TIE'

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                
                # Try multiple parsing strategies to reduce unnecessary TIEs
                winner = None
                
                # Strategy 1: Look for "More aligned: X" pattern
                m = re.search(r"More aligned:\s*(A|B|Tie)", content, re.IGNORECASE)
                if m:
                    winner = m.group(1).upper()
                
                # Strategy 2: Look for standalone A/B/Tie at the end
                if not winner:
                    lines = content.strip().split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if re.match(r'^(A|B|Tie)$', line, re.IGNORECASE):
                            winner = line.upper()
                            break
                
                # Strategy 3: Look for any A/B/Tie in the response
                if not winner:
                    matches = re.findall(r'\b(A|B|Tie)\b', content, re.IGNORECASE)
                    if matches:
                        winner = matches[-1].upper()  # Take the last occurrence
                
                # Strategy 4: Last resort - check last word
                if not winner:
                    tail = content.strip().split()[-1].strip().upper()
                    if tail in {'A', 'B', 'TIE'}:
                        winner = tail
                
                # Normalize TIE/Tie to TIE
                if winner and winner.upper() == 'TIE':
                    winner = 'TIE'
                    
                if winner in {'A', 'B', 'TIE'}:
                    judgment = winner
                    break
                else:
                    # Print debug info for failed parsing
                    print(f"⚠️ Failed to parse judgment from: {content[:100]}...")
                    if attempt == max_retries - 1:
                        judgment = 'TIE'  # Default to TIE only as last resort
            except Exception as e:
                print(f"⚠️ API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    judgment = 'TIE'
                else:
                    time.sleep(sleep_time * 2)
                    continue

        out = dict(item)
        out.update({
            'judgment': judgment,
            'model': model,
            'judgment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        results.append(out)

        # Save intermediate results every 50 judgments
        if (i + 1) % 50 == 0:
            # Save to main output file for resumability
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open('w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"💾 Saved progress: {len(results)} total judgments")

        time.sleep(sleep_time)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    a_wins = sum(1 for r in results if r['judgment'] == 'A')
    b_wins = sum(1 for r in results if r['judgment'] == 'B')
    ties = sum(1 for r in results if r['judgment'] == 'TIE')
    return a_wins, b_wins, ties


def analyze_results(judged_jsonl: Path) -> Dict[str, Any]:
    with judged_jsonl.open('r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    base_wins = 0
    rps_wins = 0
    ties = 0
    for r in rows:
        j = r['judgment']
        if j == 'A':
            base_wins += 1 if r.get('a_is_baseline') else 0
            rps_wins += 0 if r.get('a_is_baseline') else 1
        elif j == 'B':
            base_wins += 0 if r.get('a_is_baseline') else 1
            rps_wins += 1 if r.get('a_is_baseline') else 0
        else:
            ties += 1

    total = len(rows)
    win_rate = rps_wins / (rps_wins + base_wins) if (rps_wins + base_wins) > 0 else 0.5
    return {
        'total': total,
        'baseline_wins': base_wins,
        'rps_wins': rps_wins,
        'ties': ties,
        'rps_win_rate': win_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare Ultra DPO baseline vs RPS best responses'
    )
    parser.add_argument('--baseline_dir', type=str, required=True, 
                       help='Directory containing baseline CSVs (dpo_responses_v{X}_merged.csv)')
    parser.add_argument('--rps_dir', type=str, required=True, 
                       help='Directory containing RPS CSVs (v{X}_best_response.csv)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to write comparison artifacts')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='OpenAI model for judging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--directions', type=str, nargs='+', 
                       default=[f"v{i}" for i in range(3, 11)],
                       help='Directions to process (default: v3-v10)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process per direction (for testing/resuming)')
    parser.add_argument('--max_tokens', type=int, default=256,
                       help='Maximum tokens for GPT response (default: 256)')
    args = parser.parse_args()

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        return

    set_seed(args.seed)

    baseline_dir = Path(args.baseline_dir)
    rps_dir = Path(args.rps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_base = sorted([p for p in baseline_dir.glob('**/*.csv')])
    all_rps = sorted([p for p in rps_dir.glob('**/*.csv')])

    summary_rows = []

    print(f"📂 Baseline files: {len(all_base)} | RPS files: {len(all_rps)}")
    print(f"🎯 Processing directions: {args.directions}")

    for direction in args.directions:
        print(f"\n{'='*60}\n🎯 Processing {direction}\n{'='*60}")
        
        # Find files for this direction
        base_files = _find_direction_csv_ultra(all_base, direction)
        rps_files = _find_direction_csv_ultra(all_rps, direction)
        
        print(f"Found baseline files: {[f.name for f in base_files]}")
        print(f"Found RPS files: {[f.name for f in rps_files]}")
        
        if not base_files:
            print(f"❌ No baseline CSVs for {direction}")
            continue
        if not rps_files:
            print(f"❌ No RPS CSVs for {direction}")
            continue

        # Load and concatenate in case there are multiple files per direction
        base_dfs = [df for df in (_load_best_df_ultra(p) for p in base_files) if df is not None]
        rps_dfs = [df for df in (_load_best_df_ultra(p) for p in rps_files) if df is not None]
        
        if not base_dfs or not rps_dfs:
            print(f"⚠️ Skipping {direction} due to unreadable files")
            continue

        base_df = pd.concat(base_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])
        rps_df = pd.concat(rps_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])

        print(f"📊 Baseline DF: {len(base_df)} rows, RPS DF: {len(rps_df)} rows")

        # Generate pairwise comparisons
        pair_jsonl = out_dir / f"{direction}_pairwise.jsonl"
        n_pairs = generate_pairwise_jsonl_for_gpt_judging(
            baseline_df=base_df, 
            rps_df=rps_df, 
            direction_name=direction, 
            out_jsonl=pair_jsonl, 
            seed=args.seed
        )
        
        if n_pairs == 0:
            print(f"❌ No matching prompt_id between baseline and RPS for {direction}")
            continue
        print(f"✅ Generated {n_pairs} pairs: {pair_jsonl}")

        # Run GPT judging
        judged_jsonl = out_dir / f"{direction}_judged.jsonl"
        a_wins, b_wins, ties = run_gpt_judging(pair_jsonl, judged_jsonl, model=args.model, max_samples=args.max_samples, max_tokens=args.max_tokens)
        print(f"📊 Raw A/B/TIE: {a_wins}/{b_wins}/{ties}")

        # Analyze results
        stats = analyze_results(judged_jsonl)
        stats['direction'] = direction
        summary_rows.append(stats)
        print(
            f"   Baseline wins: {stats['baseline_wins']} | RPS wins: {stats['rps_wins']} | "
            f"Ties: {stats['ties']} | RPS win-rate (excl. ties): {stats['rps_win_rate']:.3f}"
        )

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "ultra_comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n📈 Summary saved: {summary_csv}")
        
        # Print overall statistics
        total_baseline_wins = summary_df['baseline_wins'].sum()
        total_rps_wins = summary_df['rps_wins'].sum()
        total_ties = summary_df['ties'].sum()
        overall_rps_win_rate = total_rps_wins / (total_rps_wins + total_baseline_wins) if (total_rps_wins + total_baseline_wins) > 0 else 0.5
        
        print(f"\n🏆 Overall Results:")
        print(f"   Total comparisons: {total_baseline_wins + total_rps_wins + total_ties}")
        print(f"   Baseline wins: {total_baseline_wins}")
        print(f"   RPS wins: {total_rps_wins}")
        print(f"   Ties: {total_ties}")
        print(f"   RPS win-rate (excl. ties): {overall_rps_win_rate:.3f}")
    else:
        print("\n⚠️ No results to summarize")


if __name__ == "__main__":
    main()
