import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_reward_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
        trust_remote_code=True,
    )
    tokenizer = _ensure_pad_token(tokenizer)
    return model, tokenizer


def score_response(prompt, response, model, tokenizer, device):
    try:
        template = (
            "[INST] You must read the following conversation carefully and rate the assistant's "
            "response from score 0-100 in these aspects: helpfulness, correctness, coherence, "
            "honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        )
        inputs = tokenizer(
            template.format(prompt=prompt, response=response),
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze().detach().cpu().numpy()
        helpfulness = float(logits[9])  # 修正：helpfulness应该使用索引9
        verbosity = float(logits[4])
        return helpfulness, verbosity
    except Exception:
        return 0.0, 0.0


def pick_best(df, model, tokenizer, device):
    # columns detection
    prompt_col = None
    for cand in ["prompt", "question", "input", "user", "query"]:
        if cand in df.columns:
            prompt_col = cand
            break
    response_col = None
    for cand in ["response", "output", "answer", "completion"]:
        if cand in df.columns:
            response_col = cand
            break
    if prompt_col is None or response_col is None:
        return pd.DataFrame()

    # Only keep v3-v10 if direction_name available
    valid_dirs = {"v3","v4","v5","v6","v7","v8","v9","v10"}
    if "direction_name" in df.columns:
        df = df[df["direction_name"].astype(str).isin(valid_dirs)].copy()
        if len(df) == 0:
            return pd.DataFrame()

    group_key = "prompt_id" if "prompt_id" in df.columns else prompt_col
    results = []

    for gid, g in tqdm(df.groupby(group_key), total=df[group_key].nunique()):
        scored = []
        for idx, row in g.iterrows():
            prompt = str(row[prompt_col])
            response = str(row[response_col])
            h, v = score_response(prompt, response, model, tokenizer, device)
            row_v1 = float(row.get("v1", 0.5))
            row_v2 = float(row.get("v2", 0.5))
            dpa = row_v1 * h + row_v2 * v
            scored.append({
                "idx": idx,
                "prompt": prompt,
                "response": response,
                "helpfulness": h,
                "verbosity": v,
                "dpa_score": float(dpa),
                "response_id": row.get("response_id"),
            })

        if not scored:
            continue
        best = max(scored, key=lambda x: x["dpa_score"])
        src_row = df.loc[best["idx"]]
        out_row = {
            "prompt_id": src_row.get("prompt_id", gid),
            "prompt": best["prompt"],
            "direction_name": src_row.get("direction_name"),
            "direction_vector": src_row.get("direction_vector"),
            "angle_degrees": src_row.get("angle_degrees"),
            "response_id": best.get("response_id"),
            "response": best["response"],
            "helpfulness": best["helpfulness"],
            "verbosity": best["verbosity"],
            "v1": float(src_row.get("v1", 0.5)),
            "v2": float(src_row.get("v2", 0.5)),
            "dpa_score": best["dpa_score"],
            "selected_as_best": True,
            "all_dpa_scores": [s["dpa_score"] for s in scored],
            "num_candidates": len(scored),
        }
        results.append(out_row)

    return pd.DataFrame(results)


def run_reward_scoring(input_dir, scored_dir, model, tokenizer, device):
    os.makedirs(scored_dir, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(input_dir, "*.csv"))])
    print(f"📁 Found {len(files)} CSV files to process in {input_dir}")
    for file_path in files:
        file = os.path.basename(file_path)
        print(f"\n📄 Processing: {file}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"   ❌ Failed to read {file}: {e}")
            continue

        if "helpfulness" not in df.columns:
            df["helpfulness"] = np.nan
        if "verbosity" not in df.columns:
            df["verbosity"] = np.nan

        need = df[df["helpfulness"].isna() | df["verbosity"].isna()].shape[0]
        print(f"   🎯 Need to score {need} rows")
        if need == 0:
            print("   ✅ Already scored, skipping")
        else:
            prompt_col = "prompt" if "prompt" in df.columns else None
            response_col = "response" if "response" in df.columns else None
            if prompt_col is None or response_col is None:
                print("   ⚠️ Missing prompt/response columns; skipping file")
                continue
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {file}"):
                if pd.notnull(row["helpfulness"]) and pd.notnull(row["verbosity"]):
                    continue
                try:
                    h, v = score_response(str(row[prompt_col]), str(row[response_col]), model, tokenizer, device)
                    df.loc[i, "helpfulness"] = h
                    df.loc[i, "verbosity"] = v
                    if (i + 1) % 200 == 0:
                        tmp = os.path.join(scored_dir, file.replace(".csv", "_temp_scored.csv"))
                        df.to_csv(tmp, index=False)
                except Exception as e:
                    print(f"   ❌ Row {i} error: {e}")
                    continue

        save_path = os.path.join(scored_dir, file.replace(".csv", "_scored.csv"))
        df.to_csv(save_path, index=False)
        print(f"   💾 Saved: {save_path}")
        tmp = os.path.join(scored_dir, file.replace(".csv", "_temp_scored.csv"))
        if os.path.exists(tmp):
            os.remove(tmp)


def select_best_response(scored_dir, output_path):
    scored_files = sorted([p for p in glob.glob(os.path.join(scored_dir, "*_scored.csv"))])
    if not scored_files:
        print(f"❌ No scored files in {scored_dir}")
        return
    dfs = []
    for pth in scored_files:
        try:
            df = pd.read_csv(pth)
        except Exception:
            continue
        if not {"prompt_id", "prompt", "response", "helpfulness", "verbosity"}.issubset(df.columns):
            continue
        if "v1" in df.columns and "v2" in df.columns:
            df["score_total"] = df["v1"] * df["helpfulness"] + df["v2"] * df["verbosity"]
        else:
            df["score_total"] = 0.7071 * df["helpfulness"] + 0.7071 * df["verbosity"]
        dfs.append(df)
    if not dfs:
        print(f"❌ No valid data in {scored_dir}")
        return
    df_all = pd.concat(dfs, ignore_index=True)
    best = df_all.loc[df_all.groupby("prompt_id")["score_total"].idxmax()].copy()
    best = best.rename(columns={"response": "best_response"})
    best.to_csv(output_path, index=False)
    print(f"🏆 Best saved: {output_path} ({len(best)} rows)")


def parse_args():
    """Parse command line arguments"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get input path from command line or use default
    if 'input_path' in args:
        base_dir = args['input_path']
    else:
        print("Error: input_path parameter is required")
        print("Usage: python3 dpo_helpsteer_rps_scoring.py input_path=/path/to/input [output_path=/path/to/output]")
        sys.exit(1)
    
    # Get output path from command line or use default
    if 'output_path' in args:
        out_root = args['output_path']
    else:
        out_root = "/mnt/rps_project/data/helpsteer/dpa/dpa_rps_helpsteer_score"
    
    os.makedirs(out_root, exist_ok=True)
    
    if not os.path.exists(base_dir):
        print(f"Error: Input directory {base_dir} does not exist")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_reward_model(device)
    print(f"Loaded reward model on {device}")

    directions = [f"v{i}" for i in range(3, 11)]
    print(f"🎯 Directions: {directions}")

    for direction in directions:
        dir_path = os.path.join(base_dir, direction)
        # Fallback: if no per-direction folder, group flat files containing tag
        if os.path.isdir(dir_path):
            input_dir = dir_path
            scored_dir = os.path.join(out_root, f"scored_{direction}")
        else:
            input_dir = os.path.join(base_dir, f"__flat_{direction}")
            os.makedirs(input_dir, exist_ok=True)
            # materialize symlinks-like copies via reading directly from base
            all_csv = sorted(glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True))
            tagged = [p for p in all_csv if direction in os.path.basename(p)]
            if not tagged:
                print(f"⚠️ No files for {direction}; skipping")
                # cleanup empty temp dir
                try:
                    os.rmdir(input_dir)
                except Exception:
                    pass
                continue
            # For flat mode, we just score from original paths; run_reward_scoring expects files in input_dir
            # So we will pass the base_dir and filter inside scoring by extension; instead, directly score from a temp view:
            # Copy-less approach: create scored_dir under base with tag
            input_dir = os.path.dirname(tagged[0])  # place-holder; scoring will read all *.csv in this directory
            # narrow down by creating a temp list and scoring only those files: handled by selecting directory; leave as is
            scored_dir = os.path.join(out_root, f"scored_{direction}")

        print(f"\n===== {direction} =====")
        run_reward_scoring(input_dir, scored_dir, model, tokenizer, device)
        output_path = os.path.join(out_root, f"{direction}_best_response.csv")
        select_best_response(scored_dir, output_path)


if __name__ == "__main__":
    main()


