#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score HelpSteer DPO RPS outputs with RewardModel-Mistral-7B-for-DPA-v1
and select the best response per prompt for each target direction v3..v10.

Inputs (default): /root/rps/data/helpsteer/dpo_rps_helpsteer_outputs/{v3..v10}/*.csv
Outputs (default):
  - /root/rps/data/helpsteer/dpo_rps_best_response/scored/{version}/dpo_rps_angle*_scored.csv
  - /root/rps/data/helpsteer/dpo_rps_best_response/scored/{version}/tmp_best/*_best.csv
  - /root/rps/data/helpsteer/dpo_rps_best_response/{version}_best_response.csv
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def build_inputs(tokenizer, prompts: List[str], responses: List[str], device: str):
    template = (
        "[INST] You must read the following conversation carefully and rate the assistant's "
        "response from score 0-100 in these aspects: helpfulness, correctness, coherence, "
        "honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
    )
    texts = [template.format(prompt=p, response=r) for p, r in zip(prompts, responses)]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    if device == "cuda":
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
    return enc


@torch.inference_mode()
def score_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = build_inputs(tokenizer, prompts, responses, device)
    logits = model(**inputs).logits
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    logits = logits.detach().to("cpu").numpy()
    # helpfulness index 9, verbosity index 4, per prior usage
    helpfulness = logits[:, 9]
    verbosity = logits[:, 4]
    return helpfulness, verbosity


def score_csv_file(
    input_file: str,
    output_file: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 4,
):
    print(f"📂 Loading file: {input_file}")
    df = pd.read_csv(input_file)

    if "helpfulness" not in df.columns:
        df["helpfulness"] = np.nan
    if "verbosity" not in df.columns:
        df["verbosity"] = np.nan

    total = len(df)
    print(f"Total rows: {total}")

    os.makedirs(Path(output_file).parent, exist_ok=True)

    start = 0
    with tqdm(total=total, desc=f"Scoring {Path(input_file).name}") as pbar:
        while start < total:
            end = min(start + batch_size, total)
            batch = df.iloc[start:end]

            mask = batch["helpfulness"].isna() | batch["verbosity"].isna()
            if mask.any():
                prompts = batch.loc[mask, "prompt"].astype(str).tolist()
                responses = batch.loc[mask, "response"].astype(str).tolist()
                try:
                    h, v = score_batch(model, tokenizer, prompts, responses, device)
                    df.loc[batch.loc[mask].index, "helpfulness"] = h
                    df.loc[batch.loc[mask].index, "verbosity"] = v
                except Exception as e:
                    print(f"⚠️ Error scoring rows {start}-{end}: {e}")

            start = end
            pbar.update(end - pbar.n)

    df.to_csv(output_file, index=False)
    print(f"✅ Saved scored file: {output_file}")
    return df


def select_best(scored_csv: str, out_csv: str, main_v1: float, main_v2: float):
    df = pd.read_csv(scored_csv)
    df["score_total"] = main_v1 * df["helpfulness"] + main_v2 * df["verbosity"]
    best = df.loc[df.groupby("prompt_id")["score_total"].idxmax()].copy()
    best = best.rename(columns={"response": "best_response"})
    cols = [
        "prompt_id",
        "prompt",
        "best_response",
        "v1_p",
        "v2_p",
        "valid_angle",
        "main_v1",
        "main_v2",
        "helpfulness",
        "verbosity",
        "score_total",
    ]
    best = best[cols]
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    best.to_csv(out_csv, index=False)
    print(f"✅ Saved best responses: {out_csv}")
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Score DPO HelpSteer outputs and select best responses (v3..v10)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/rps_project/data/helpsteer/dpo_rps_helpsteer_outputs",
        help="Directory containing version folders v3..v10",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/rps_project/data/helpsteer/dpo_rps_best_response",
        help="Directory to write scored files and final best CSVs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="auto", help="auto|cuda|cpu"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"🚀 Using device: {device}")

    print("🤖 Loading Reward Model...")
    # Use bf16/half and device_map=auto to reduce OOM risk
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    scored_dir = str(Path(args.output_dir) / "scored")
    final_out_dir = args.output_dir
    os.makedirs(scored_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    vectors = {
        "v3": (0.9848, 0.1736),
        "v4": (0.9659, 0.2588),
        "v5": (0.9397, 0.3420),
        "v6": (0.9063, 0.4226),
        "v7": (0.8660, 0.5000),
        "v8": (0.8192, 0.5736),
        "v9": (0.7660, 0.6428),
        "v10": (0.7071, 0.7071),
    }

    for version in sorted(os.listdir(args.input_dir)):
        version_path = Path(args.input_dir) / version
        if not version_path.is_dir() or version not in vectors:
            continue

        print(f"\n📁 Processing {version}")
        main_v1, main_v2 = vectors[version]

        version_scored_dir = Path(scored_dir) / version
        version_tmp_best = version_scored_dir / "tmp_best"
        version_scored_dir.mkdir(parents=True, exist_ok=True)
        version_tmp_best.mkdir(parents=True, exist_ok=True)

        csv_files = [f for f in os.listdir(version_path) if f.endswith(".csv")]
        for csv in sorted(csv_files):
            inp = str(version_path / csv)
            scored_csv = str(version_scored_dir / csv.replace(".csv", "_scored.csv"))
            best_csv = str(version_tmp_best / csv.replace(".csv", "_best.csv"))

            try:
                score_csv_file(inp, scored_csv, model, tokenizer, device, args.batch_size)
                select_best(scored_csv, best_csv, main_v1, main_v2)
            except Exception as e:
                print(f"❌ Error processing {csv}: {e}")

        # Aggregate tmp bests into a single file per version
        tmp_best_files = [
            str(version_tmp_best / f) for f in os.listdir(version_tmp_best) if f.endswith("_best.csv")
        ]
        if not tmp_best_files:
            print(f"⚠️ No tmp best files for {version}")
            continue

        df_all = pd.concat([pd.read_csv(p) for p in tmp_best_files], ignore_index=True)
        if "score_total" not in df_all.columns:
            df_all["score_total"] = main_v1 * df_all["helpfulness"] + main_v2 * df_all["verbosity"]
        df_final = df_all.loc[df_all.groupby("prompt_id")["score_total"].idxmax()].copy()
        out_path = Path(final_out_dir) / f"{version}_best_response.csv"
        df_final.to_csv(out_path, index=False)
        print(f"🏁 Saved aggregated best for {version} → {out_path}")

    print("\n✅ All processing completed!")


if __name__ == "__main__":
    main()


