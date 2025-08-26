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


def _find_direction_csv(files: List[Path], direction: str) -> List[Path]:
    pat = re.compile(fr"\b{re.escape(direction)}\b", re.IGNORECASE)
    matched = [p for p in files if pat.search(p.stem) or pat.search(p.name)]
    return matched


def _load_best_df(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
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
        return df
    except Exception:
        return None


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

    left_cols = ['prompt_id', 'prompt']
    if 'v1' in baseline_df.columns and 'v2' in baseline_df.columns:
        left_cols += ['v1', 'v2']

    right_cols = ['prompt_id', 'prompt']
    if 'v1' in rps_df.columns and 'v2' in rps_df.columns:
        right_cols += ['v1', 'v2']

    merged = pd.merge(
        baseline_df[left_cols + [base_resp_col]],
        rps_df[right_cols + [rps_resp_col]],
        on=['prompt_id', 'prompt'],
        suffixes=('_baseline', '_rps')
    )

    if len(merged) == 0:
        return 0

    pairwise: List[Dict[str, Any]] = []
    for i, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Pairwise {direction_name}"):
        prompt: str = row['prompt']
        v1 = float(row['v1_baseline'] if 'v1_baseline' in row else row.get('v1_rps', 0.7071))
        v2 = float(row['v2_baseline' if 'v2_baseline' in row else 'v2_rps'] if ('v2_baseline' in row or 'v2_rps' in row) else 0.7071)

        if random.random() < 0.5:
            response_a = row[f'{base_resp_col}_baseline']
            response_b = row[f'{rps_resp_col}_rps']
            a_is_baseline = True
        else:
            response_a = row[f'{rps_resp_col}_rps']
            response_b = row[f'{base_resp_col}_baseline']
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


def run_gpt_judging(input_jsonl: Path, output_jsonl: Path, model: str = "gpt-4o-mini", sleep_time: float = 0.7, max_retries: int = 3) -> Tuple[int, int, int]:
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package is required. pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key
    api_base = os.environ.get("OPENAI_API_BASE")
    if api_base:
        openai.api_base = api_base

    with input_jsonl.open('r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    results: List[Dict[str, Any]] = []

    for i, item in tqdm(list(enumerate(data)), total=len(data), desc=f"Judging {model}"):
        prompt = build_appendix_a2_prompt(
            prompt=item['prompt'],
            response_a=item['response_a'],
            response_b=item['response_b'],
            v1=float(item.get('v1', 0.7071)),
            v2=float(item.get('v2', 0.7071)),
        )
        judgment: str = 'Tie'

        for attempt in range(max_retries):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=32,
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
                    time.sleep(sleep_time * 2)
                    continue

        out = dict(item)
        out.update({
            'judgment': judgment,
            'model': model,
            'judgment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        results.append(out)

        if (i + 1) % 50 == 0:
            tmp = output_jsonl.with_name(output_jsonl.stem + f"_temp_{i+1}.jsonl")
            with tmp.open('w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True, help='Directory containing baseline CSVs')
    parser.add_argument('--rps_dir', type=str, required=True, help='Directory containing RPS CSVs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write comparison artifacts')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    baseline_dir = Path(args.baseline_dir)
    rps_dir = Path(args.rps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_base = sorted([p for p in baseline_dir.glob('**/*.csv')])
    all_rps = sorted([p for p in rps_dir.glob('**/*.csv')])

    directions = [f"v{i}" for i in range(3, 11)]
    summary_rows = []

    print(f"📂 Baseline files: {len(all_base)} | RPS files: {len(all_rps)}")

    for direction in directions:
        print(f"\n{'='*60}\n🎯 Processing {direction}\n{'='*60}")
        base_files = _find_direction_csv(all_base, direction)
        rps_files = _find_direction_csv(all_rps, direction)
        if not base_files:
            print(f"❌ No baseline CSVs for {direction}")
            continue
        if not rps_files:
            print(f"❌ No RPS CSVs for {direction}")
            continue

        # Load and concatenate in case there are multiple files per direction
        base_dfs = [df for df in (_load_best_df(p) for p in base_files) if df is not None]
        rps_dfs = [df for df in (_load_best_df(p) for p in rps_files) if df is not None]
        if not base_dfs or not rps_dfs:
            print(f"⚠️ Skipping {direction} due to unreadable files")
            continue

        base_df = pd.concat(base_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])
        rps_df = pd.concat(rps_dfs, ignore_index=True).drop_duplicates(subset=['prompt_id'])

        pair_jsonl = out_dir / f"{direction}_pairwise.jsonl"
        n_pairs = generate_pairwise_jsonl_for_gpt_judging(baseline_df=base_df, rps_df=rps_df, direction_name=direction, out_jsonl=pair_jsonl, seed=args.seed)
        if n_pairs == 0:
            print(f"❌ No matching prompt_id between baseline and RPS for {direction}")
            continue
        print(f"✅ Generated {n_pairs} pairs: {pair_jsonl}")

        judged_jsonl = out_dir / f"{direction}_judged.jsonl"
        a_wins, b_wins, ties = run_gpt_judging(pair_jsonl, judged_jsonl, model=args.model)
        print(f"📊 Raw A/B/TIE: {a_wins}/{b_wins}/{ties}")

        stats = analyze_results(judged_jsonl)
        stats['direction'] = direction
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
    else:
        print("\n⚠️ No results to summarize")


if __name__ == "__main__":
    main()


