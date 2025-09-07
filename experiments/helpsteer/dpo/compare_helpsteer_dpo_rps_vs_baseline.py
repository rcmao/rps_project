#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HelpSteer DPO: RPS vs Baseline comparison

This script mirrors the flow of dpo_baseline_comparison.py, but compares
HelpSteer DPO baseline best responses against RPS best responses, using
the Appendix A.2 judge prompt from the paper (preference-aligned A/B/Tie).

Pipeline per direction (v3..v10):
1) Load baseline and RPS CSVs and inner-join by prompt_id
2) Randomize A/B assignment and write pairwise JSONL
3) Judge pairs with GPT using weights (v1, v2) from the rows
4) Analyze win/tie rates and write summary CSV

Usage:
  python compare_helpsteer_dpo_rps_vs_baseline.py \
      --baseline_dir /path/to/dpo_helpsteer_baseline_scoring \
      --rps_dir /path/to/dpo_rps_best_resposne \
      --output_dir /path/to/compare_outputs \
      --model gpt-4o-mini \
      --max_workers 16 \
      --skip_existing

Performance Options:
  --max_workers: Number of parallel workers for GPT API calls (default: 8)
  --skip_existing: Skip directions with existing judged files for resume capability

Environment:
  Uses OpenRouter API. Set OPENROUTER_API_KEY or modify the hardcoded key in the script.
"""

import os
import re
import json
import time
import argparse
import random
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def _find_direction_csv(files: List[Path], direction: str) -> List[Path]:
    # 更灵活的模式匹配，适应不同的文件命名约定
    # 匹配: baseline_responses_v3.csv, v3_best_response.csv 等
    pat = re.compile(fr"(?:^|_){re.escape(direction)}(?:_|$|\.)", re.IGNORECASE)
    matched = [p for p in files if pat.search(p.stem) or pat.search(p.name)]
    return matched


def _load_best_df(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        # Use more efficient reading with specific dtypes
        df = pd.read_csv(csv_path, dtype={'prompt_id': 'string'})
        # normalize likely columns
        # expected at least: prompt_id, prompt, response/best_response, v1, v2
        # unify response column name to 'best_response'
        if 'best_response' not in df.columns:
            if 'response' in df.columns:
                df = df.rename(columns={'response': 'best_response'})
        # ensure required columns exist
        needed = {'prompt_id', 'prompt', 'best_response'}
        if not needed.issubset(df.columns):
            return None
        
        # Optimize memory usage by converting to categories for repeated strings
        if 'prompt_id' in df.columns:
            df['prompt_id'] = df['prompt_id'].astype('category')
            
        return df
    except Exception:
        return None


def _load_multiple_csvs_parallel(csv_paths: List[Path]) -> List[pd.DataFrame]:
    """Load multiple CSV files in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {executor.submit(_load_best_df, path): path for path in csv_paths}
        results = []
        for future in as_completed(future_to_path):
            df = future.result()
            if df is not None:
                results.append(df)
    return results


def generate_pairwise_jsonl_for_gpt_judging(
    baseline_df: pd.DataFrame,
    rps_df: pd.DataFrame,
    direction_name: str,
    out_jsonl: Path,
    seed: int = 42,
) -> int:
    set_seed(seed)

    base_resp_col = 'best_response'
    rps_resp_col = 'best_response'

    # Optimize column selection
    left_cols = ['prompt_id', 'prompt', base_resp_col]
    right_cols = ['prompt_id', 'prompt', rps_resp_col]
    
    # Add v1, v2 columns if they exist
    if 'v1' in baseline_df.columns and 'v2' in baseline_df.columns:
        left_cols += ['v1', 'v2']
    if 'v1' in rps_df.columns and 'v2' in rps_df.columns:
        right_cols += ['v1', 'v2']

    # Use more efficient merge with inner join
    merged = pd.merge(
        baseline_df[left_cols],
        rps_df[right_cols],
        on=['prompt_id', 'prompt'],
        suffixes=('_baseline', '_rps'),
        how='inner'
    )

    if len(merged) == 0:
        return 0

    # Vectorized operations for better performance
    n_rows = len(merged)
    random_choices = [random.random() < 0.5 for _ in range(n_rows)]
    
    # Pre-compute v1, v2 values
    v1_values = []
    v2_values = []
    for _, row in merged.iterrows():
        v1 = float(row.get('v1_baseline', row.get('v1_rps', 0.7071)))
        v2 = float(row.get('v2_baseline', row.get('v2_rps', 0.7071)))
        v1_values.append(v1)
        v2_values.append(v2)

    # Build pairwise data more efficiently
    pairwise = []
    base_col_name = f'{base_resp_col}_baseline'
    rps_col_name = f'{rps_resp_col}_rps'
    
    for i, (_, row) in enumerate(tqdm(merged.iterrows(), total=n_rows, desc=f"Pairwise {direction_name}")):
        if random_choices[i]:
            response_a = row[base_col_name]
            response_b = row[rps_col_name]
            a_is_baseline = True
        else:
            response_a = row[rps_col_name]
            response_b = row[base_col_name]
            a_is_baseline = False

        item = {
            'id': f'{direction_name}_{i}',
            'prompt_id': str(row['prompt_id']),  # Ensure string type
            'prompt': str(row['prompt']),
            'response_a': str(response_a),
            'response_b': str(response_b),
            'a_is_baseline': a_is_baseline,
            'direction': direction_name,
            'v1': v1_values[i],
            'v2': v2_values[i],
        }
        pairwise.append(item)

    # Write JSONL more efficiently
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open('w', encoding='utf-8') as f:
        for item in pairwise:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

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


def _judge_single_item(item: Dict[str, Any], model: str, max_retries: int = 3) -> Dict[str, Any]:
    """Judge a single item with retries"""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. pip install openai") from e

    # Configure OpenRouter client
    # You can also set OPENROUTER_API_KEY environment variable instead of hardcoding
    api_key = os.environ.get("OPENROUTER_API_KEY", "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")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

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
                temperature=0.1,  # 从0.0改为0.1，增加一点随机性
                max_tokens=256,  # 从32改为150
            )
            content = resp.choices[0].message.content.strip()
            # parse final line that should contain More aligned: A|B|Tie
            # fallback: search for standalone A/B/TIE
            m = re.search(r"More aligned:\s*(A|B|Tie)", content, re.IGNORECASE)
            if m:
                winner = m.group(1).upper()
            else:
                # fallback: last token A|B|TIE
                tail = content.strip().split()[-1].strip().upper()
                winner = 'TIE' if tail not in {'A', 'B'} else tail
            if winner in {'A', 'B', 'TIE'}:
                judgment = winner
                break
        except Exception:
            if attempt == max_retries - 1:
                judgment = 'TIE'
            else:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                continue

    out = dict(item)
    out.update({
        'judgment': judgment,
        'model': model,
        'judgment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    return out


def run_gpt_judging(input_jsonl: Path, output_jsonl: Path, model: str = "gpt-4o-mini", 
                   max_workers: int = 8, max_retries: int = 3, batch_size: int = 50) -> Tuple[int, int, int]:
    """Optimized GPT judging with parallel processing"""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. pip install openai") from e

    # OpenRouter configuration is handled in _judge_single_item function

    # Load data
    with input_jsonl.open('r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    
    def save_batch(batch_results: List[Dict[str, Any]], batch_num: int):
        """Save intermediate results"""
        tmp = output_jsonl.with_name(output_jsonl.stem + f"_temp_batch_{batch_num}.jsonl")
        with tmp.open('w', encoding='utf-8') as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Process in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(_judge_single_item, item, model, max_retries): (i, item)
            for i, item in enumerate(data)
        }
        
        # Process completed tasks with progress bar
        batch_results = []
        batch_num = 0
        
        with tqdm(total=len(data), desc=f"Judging {model} (parallel)") as pbar:
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                        batch_results.append(result)
                    
                    # Save intermediate batches
                    if len(batch_results) >= batch_size:
                        save_batch(batch_results, batch_num)
                        batch_results = []
                        batch_num += 1
                        
                except Exception as e:
                    # Handle failed tasks
                    i, item = future_to_item[future]
                    fallback_result = dict(item)
                    fallback_result.update({
                        'judgment': 'TIE',
                        'model': model,
                        'judgment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'error': str(e)
                    })
                    with results_lock:
                        results.append(fallback_result)
                
                pbar.update(1)
        
        # Save final batch
        if batch_results:
            save_batch(batch_results, batch_num)

    # Sort results by original order (based on id or index)
    results.sort(key=lambda x: data.index(next(item for item in data if item['id'] == x['id'])))

    # Save final results
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True, help='Directory containing baseline CSVs')
    parser.add_argument('--rps_dir', type=str, required=True, help='Directory containing RPS CSVs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write comparison artifacts')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_workers', type=int, default=8, help='Max parallel workers for GPT judging')
    parser.add_argument('--skip_existing', action='store_true', help='Skip directions with existing judged files')
    args = parser.parse_args()

    set_seed(args.seed)

    baseline_dir = Path(args.baseline_dir)
    rps_dir = Path(args.rps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all CSV files in parallel
    print("🔍 Discovering CSV files...")
    all_base = sorted([p for p in baseline_dir.glob('**/*.csv')])
    all_rps = sorted([p for p in rps_dir.glob('**/*.csv')])

    directions = [f"v{i}" for i in range(3, 11)]
    summary_rows = []

    print(f"📂 Baseline files: {len(all_base)} | RPS files: {len(all_rps)}")

    # Cache for loaded DataFrames to avoid reloading
    df_cache = {}

    for direction in directions:
        print(f"\n{'='*60}\n🎯 Processing {direction}\n{'='*60}")
        
        # Check if results already exist
        judged_jsonl = out_dir / f"{direction}_judged.jsonl"
        if args.skip_existing and judged_jsonl.exists():
            print(f"⏭️ Skipping {direction} (results already exist)")
            try:
                stats = analyze_results(judged_jsonl)
                stats['direction'] = direction
                summary_rows.append(stats)
                print(f"   Loaded existing results: {stats['rps_win_rate']:.3f} win rate")
            except Exception as e:
                print(f"⚠️ Error loading existing results: {e}")
            continue
        
        base_files = _find_direction_csv(all_base, direction)
        rps_files = _find_direction_csv(all_rps, direction)
        if not base_files:
            print(f"❌ No baseline CSVs for {direction}")
            continue
        if not rps_files:
            print(f"❌ No RPS CSVs for {direction}")
            continue

        # Load DataFrames with caching
        base_key = tuple(sorted(str(p) for p in base_files))
        rps_key = tuple(sorted(str(p) for p in rps_files))
        
        if base_key not in df_cache:
            print(f"📖 Loading {len(base_files)} baseline CSV(s)...")
            base_dfs = _load_multiple_csvs_parallel(base_files)
            if not base_dfs:
                print(f"⚠️ No readable baseline files for {direction}")
                continue
            df_cache[base_key] = pd.concat(base_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])
        
        if rps_key not in df_cache:
            print(f"📖 Loading {len(rps_files)} RPS CSV(s)...")
            rps_dfs = _load_multiple_csvs_parallel(rps_files)
            if not rps_dfs:
                print(f"⚠️ No readable RPS files for {direction}")
                continue
            df_cache[rps_key] = pd.concat(rps_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])

        base_df = df_cache[base_key]
        rps_df = df_cache[rps_key]

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

        # Run parallel GPT judging
        start_time = time.time()
        a_wins, b_wins, ties = run_gpt_judging(
            pair_jsonl, 
            judged_jsonl, 
            model=args.model, 
            max_workers=args.max_workers
        )
        elapsed = time.time() - start_time
        print(f"📊 Raw A/B/TIE: {a_wins}/{b_wins}/{ties} (took {elapsed:.1f}s)")

        stats = analyze_results(judged_jsonl)
        stats['direction'] = direction
        stats['processing_time_seconds'] = elapsed
        summary_rows.append(stats)
        print(
            f"   Baseline wins: {stats['baseline_wins']} | RPS wins: {stats['rps_wins']} | "
            f"Ties: {stats['ties']} | RPS win-rate(excl. ties): {stats['rps_win_rate']:.3f}"
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n📈 Summary saved: {summary_csv}")
        
        # Print overall statistics
        total_time = sum(row.get('processing_time_seconds', 0) for row in summary_rows)
        avg_win_rate = summary_df['rps_win_rate'].mean()
        print(f"🏁 Total processing time: {total_time:.1f}s")
        print(f"📊 Average RPS win rate: {avg_win_rate:.3f}")
    else:
        print("\n⚠️ No results to summarize")


if __name__ == "__main__":
    main()


