# mistral_helpsteer_dpo_baseline_generation.py
# 使用 HelpSteer 数据集，仅生成基线响应（不加载/不使用 reward model）
# Base model: mistralai/Mistral-7B-Instruct-v0.2

import os
# Force official Hugging Face endpoint for gated model access (Colab)
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import time
import random

# 可选：镜像与缓存设置（与原脚本风格一致，可按需移除）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', '/root/.cache/huggingface')
print("🌏 已设置Hugging Face镜像(可选):", os.environ['HF_ENDPOINT'])

# 定义v3-v10的方向向量（保持与原脚本一致）
PREFERENCE_DIRECTIONS = {
    "v3": {"vector": (0.9848, 0.1736), "angle": 10},
    "v4": {"vector": (0.9659, 0.2588), "angle": 15},
    "v5": {"vector": (0.9397, 0.3420), "angle": 20},
    "v6": {"vector": (0.9063, 0.4226), "angle": 25},
    "v7": {"vector": (0.8660, 0.5000), "angle": 30},
    "v8": {"vector": (0.8192, 0.5736), "angle": 35},
    "v9": {"vector": (0.7660, 0.6428), "angle": 40},
    "v10": {"vector": (0.7071, 0.7071), "angle": 45},
}

def setup_environment():
    """设置设备与随机种子。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    seed = int(os.environ.get("SEED", "42"))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 可选：允许 TF32 加速
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return device


def _get_hf_token():
    """从环境获取 HF token。支持多种变量名。"""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("TRANSFORMERS_TOKEN")
    )


def load_generation_model(device):
    """加载用于生成的 Mistral-7B-Instruct-v0.2 与分词器。

    对于 gated 模型：
    - 优先使用官方域名 https://huggingface.co（镜像通常不支持受控鉴权）
    - 支持通过环境变量提供 token：HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN / TRANSFORMERS_TOKEN
    """
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"🤖 Loading generation model ({model_id})...")

    # 若是 gated 模型，自动切换到官方域名，除非显式允许镜像
    if "mistralai/" in model_id and os.environ.get("ALLOW_MIRROR_FOR_GATED", "0") != "1":
        if os.environ.get("HF_ENDPOINT") != "https://huggingface.co":
            print("🔐 检测到 gated 模型，已将 HF_ENDPOINT 切换为 https://huggingface.co 以支持鉴权下载。")
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    hf_token = _get_hf_token()
    if not hf_token:
        print("⚠️ 未检测到 Hugging Face token（HF_TOKEN / HUGGINGFACE_HUB_TOKEN）。对于 gated 模型，必须设置 token 并具备访问权限。")
        print("   获取方式：https://huggingface.co/settings/tokens 并在运行前设置，例如：")
        print("   os.environ['HF_TOKEN'] = 'hf_xxx'  或  使用 huggingface_hub.login(token='hf_xxx')")

    common_kwargs = dict(
        torch_dtype=torch.float16,  # 若有 bfloat16 支持也可换成 torch.bfloat16
        trust_remote_code=True,
        resume_download=True,
        low_cpu_mem_usage=True,
    )
    if hf_token:
        # transformers >= 4.33 支持 token 参数
        common_kwargs["token"] = hf_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            **common_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **({"token": hf_token} if hf_token else {}),
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Generation model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading generation model: {e}")
        print("🔄 Retrying with explicit to(device)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            **common_kwargs,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **({"token": hf_token} if hf_token else {}),
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer


def build_dpa_input(prompt, v1, v2):
    """构造带权重说明的输入（延续原脚本：把加权提示放到 user 内容中）。"""
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = (
        f"You are a helpful assistant. Your response should maximize weighted rating = "
        f"helpfulness*{h} + verbosity*{v}."
    )
    # 也可以改成使用 "system"+"user" 两条消息；这里保持与原脚本一致的单 user 形式
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]


def generate_responses_for_direction(
    prompt,
    prompt_id,
    direction_name,
    direction_info,
    model,
    tokenizer,
    device,
    num_responses=3,
):
    """为单个 prompt 在特定方向上生成多个响应（不做打分）。"""
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]

        messages = build_dpa_input(prompt, v1, v2)
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "512"))

        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        responses = []
        for i in range(num_responses):
            generated_tokens = outputs[i][input_ids.shape[1]:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "direction_name": direction_name,
                "direction_vector": f"({v1:.4f}, {v2:.4f})",
                "angle_degrees": angle,
                "response_id": i + 1,
                "response": decoded,
            })

        return responses
    except Exception as e:
        print(f"⚠️ Error generating responses for prompt {prompt_id} direction {direction_name}: {e}")
        return []


def generate_all_directions(
    prompts,
    prompt_ids,
    model,
    tokenizer,
    device,
    output_dir,
    batch_size=16,
    num_responses=3,
):
    """为所有方向生成响应（不做DPA打分与筛选）。"""
    start_time = time.time()
    all_results = []

    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"baseline_responses_{direction_name}.csv")

        # 断点续跑
        done_prompt_ids = set()
        direction_results = []

        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                done_prompt_ids = set(existing_df["prompt_id"].unique())
                print(f"🔁 Found existing results for {len(done_prompt_ids)} prompts in {direction_name}")
            except Exception as e:
                print(f"⚠️ Error loading existing file for {direction_name}: {e}")

        remaining_prompt_ids = [pid for pid in prompt_ids if pid not in done_prompt_ids]
        print(f" {direction_name}: 已处理 {len(done_prompt_ids)} 个，剩余 {len(remaining_prompt_ids)} 个")

        # 批量处理（逻辑同原脚本：逐条生成，batch_size 仅用于遍历切分）
        for start in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {direction_name} (剩余{len(remaining_prompt_ids)}个)"):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_ids = prompt_ids[start:end]

            unprocessed_indices = [i for i, pid in enumerate(batch_ids) if pid not in done_prompt_ids]
            if not unprocessed_indices:
                continue

            batch_results = []
            for i in unprocessed_indices:
                prompt = batch_prompts[i]
                prompt_id = batch_ids[i]

                responses = generate_responses_for_direction(
                    prompt,
                    prompt_id,
                    direction_name,
                    direction_info,
                    model,
                    tokenizer,
                    device,
                    num_responses,
                )
                if responses:
                    batch_results.extend(responses)

            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                if not os.path.exists(output_file):
                    df_batch.to_csv(output_file, index=False)
                else:
                    df_batch.to_csv(output_file, mode='a', header=False, index=False)

                direction_results.extend(batch_results)
                print(f"✅ Saved batch for {direction_name}, total rows: {len(direction_results)}")

        all_results.extend(direction_results)
        print(f"✅ Completed direction {direction_name}: {len(direction_results)} rows")

    elapsed = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed:.1f} seconds")
    print(f"📊 Total rows generated: {len(all_results)}")
    return all_results


def main():
    """主函数：加载 HelpSteer 数据集，仅生成基线响应并落盘。"""
    # 输出目录：优先使用环境变量 OUTPUT_DIR；否则尝试挂载并保存到 Google Drive
    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir is None:
        default_drive_dir = "/content/drive/MyDrive/baseline_sft_helpsteer"
        try:
            # 若在 Colab 环境，可用 google.colab 自动挂载
            from google.colab import drive  # type: ignore
            print("📎 Attempting to mount Google Drive at /content/drive ...")
            drive.mount('/content/drive', force_remount=False)
            os.makedirs(default_drive_dir, exist_ok=True)
            output_dir = default_drive_dir
            print(f"💾 Using Google Drive output dir: {output_dir}")
        except Exception as _:
            # 非 Colab 或挂载失败，回退到本地目录
            output_dir = "./baseline_sft_helpsteer"
            print(f"💾 Google Drive unavailable, fallback to local: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 环境
    device = setup_environment()

    # 数据集（HelpSteer）
    print("📦 Loading prompts from HelpSteer...")
    # 可选 split: "train" 或 "validation"；默认为 validation
    split = os.environ.get("HELPSTEER_SPLIT", "validation")
    num_prompts = int(os.environ.get("HELPSTEER_NUM_PROMPTS", "2000"))

    ds = load_dataset("nvidia/HelpSteer", split=split)
    prompts_series = ds["prompt"]
    prompts = prompts_series[:num_prompts]
    prompt_ids = list(range(len(prompts)))
    print(f"✅ Loaded {len(prompts)} prompts from HelpSteer ({split})!")

    # 模型
    model, tokenizer = load_generation_model(device)

    # 展示方向
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")

    # 生成
    print(f"\n🚀 Starting baseline generation for {len(prompts)} prompts across all directions...")
    _ = generate_all_directions(
        prompts=prompts,
        prompt_ids=prompt_ids,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        batch_size=int(os.environ.get("BATCH_SIZE", "32")),
        num_responses=int(os.environ.get("NUM_RESPONSES", "3")),
    )

    print(f"\n✅ All done! Results saved under {output_dir}")


if __name__ == "__main__":
    main()
