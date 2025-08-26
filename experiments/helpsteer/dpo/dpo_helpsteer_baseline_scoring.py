import os
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
        helpfulness = float(logits[0])
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
        # build output row matching expected schema when possible
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


def main():
    input_dir = "/content/drive/MyDrive/helpsteer_dpo_baseline_outputs"
    output_dir = "/content/drive/MyDrive/dpo_outputs"
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

