import os
import sys
import glob
import pandas as pd
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
        helpfulness = float(logits[9])
        verbosity = float(logits[4])
        return helpfulness, verbosity
    except Exception:
        return 0.0, 0.0


def pick_best(df, model, tokenizer, device):
    # Accept flexible schemas
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

    # Score within groups and select best, retaining aggregated fields similar to dpo_outputs schema
    group_key = "prompt_id" if "prompt_id" in df.columns else prompt_col
    results = []

    for gid, g in tqdm(df.groupby(group_key), total=df[group_key].nunique()):
        scored = []
        for idx, row in g.iterrows():
            prompt = str(row[prompt_col])
            response = str(row[response_col])
            h, v = score_response(prompt, response, model, tokenizer, device)
            
            # 只从 direction_vector 中解析 v1, v2
            direction_vector = row.get("direction_vector")
            if direction_vector is None:
                raise ValueError(f"Missing 'direction_vector' column in row {idx}")
            
            # 解析 direction_vector，假设格式为 "[v1, v2]" 或 "(v1, v2)" 或 "v1,v2"
            try:
                if isinstance(direction_vector, str):
                    # 移除括号和空格，分割字符串
                    clean_vector = direction_vector.strip("[]()").replace(" ", "")
                    v1_str, v2_str = clean_vector.split(",")
                    row_v1 = float(v1_str)
                    row_v2 = float(v2_str)
                elif isinstance(direction_vector, (list, tuple)):
                    row_v1 = float(direction_vector[0])
                    row_v2 = float(direction_vector[1])
                else:
                    raise ValueError(f"Unsupported direction_vector format: {type(direction_vector)}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Failed to parse direction_vector '{direction_vector}' in row {idx}: {e}")
            
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
        # build output row matching expected schema when possible
        src_row = df.loc[best["idx"]]
        
        # 从 direction_vector 中解析 v1, v2 用于输出
        direction_vector = src_row.get("direction_vector")
        if direction_vector is None:
            raise ValueError(f"Missing 'direction_vector' column in best response row")
        
        try:
            if isinstance(direction_vector, str):
                clean_vector = direction_vector.strip("[]()").replace(" ", "")
                v1_str, v2_str = clean_vector.split(",")
                best_v1 = float(v1_str)
                best_v2 = float(v2_str)
            elif isinstance(direction_vector, (list, tuple)):
                best_v1 = float(direction_vector[0])
                best_v2 = float(direction_vector[1])
            else:
                raise ValueError(f"Unsupported direction_vector format: {type(direction_vector)}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse direction_vector '{direction_vector}' in best response: {e}")
        
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
            "v1": float(best_v1),
            "v2": float(best_v2),
            "dpa_score": best["dpa_score"],
            "selected_as_best": True,
            "all_dpa_scores": [s["dpa_score"] for s in scored],
            "num_candidates": len(scored),
        }
        results.append(out_row)

    return pd.DataFrame(results)


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
        input_dir = args['input_path']
    else:
        print("Error: input_path parameter is required")
        print("Usage: python3 dpo_helpsteer_baseline_scoring.py input_path=/path/to/input output_path=/path/to/output")
        sys.exit(1)
    
    # Get output path from command line or use default
    if 'output_path' in args:
        output_dir = args['output_path']
    else:
        print("Error: output_path parameter is required")
        print("Usage: python3 dpo_helpsteer_baseline_scoring.py input_path=/path/to/input output_path=/path/to/output")
        sys.exit(1)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_reward_model(device)
    print(f"Loaded reward model on {device}")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print("No CSV files found in:", input_dir)
        return

    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        best = pick_best(df, model, tokenizer, device)
        if len(best) == 0:
            continue
        # stringify list field for CSV compatibility
        if "all_dpa_scores" in best.columns:
            best["all_dpa_scores"] = best["all_dpa_scores"].apply(lambda x: str(list(x)) if isinstance(x, (list, tuple)) else str(x))
        out_path = os.path.join(output_dir, os.path.basename(path))
        best.to_csv(out_path, index=False)
        print("Saved:", out_path, "rows:", len(best))


if __name__ == "__main__":
    main()

