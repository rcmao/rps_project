# Colab版本的DPO RPS扰动实验 - HelpSteer版本（仅生成，不评分）
# 模型改为 mistralai/Mistral-7B-Instruct-v0.2


# 设置内存优化
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
from google.colab import drive
from huggingface_hub import login

# 挂载Google Drive
drive.mount('/content/drive')

# 登录Hugging Face
from huggingface_hub import login
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("⚠️ No HF token found in environment. Set HF_TOKEN to access gated models.")

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ 运行环境初始化完成")

# 定义DPO的主方向向量
DPO_PREFERENCE_DIRECTIONS = {
    "v3": {"vector": (0.9848, 0.1736), "angle": 10},
    "v4": {"vector": (0.9659, 0.2588), "angle": 15}, 
    "v5": {"vector": (0.9397, 0.3420), "angle": 20},
    "v6": {"vector": (0.9063, 0.4226), "angle": 25},
    "v7": {"vector": (0.8660, 0.5000), "angle": 30},
    "v8": {"vector": (0.8192, 0.5736), "angle": 35},
    "v9": {"vector": (0.7660, 0.6428), "angle": 40},
    "v10": {"vector": (0.7071, 0.7071), "angle": 45},
}


def get_dpo_angle_perturbations(v_main, angle_range=(-40, 40), step=5, theta_max=30, top_k=5):
    """为DPO方向生成角度扰动 - 与参考脚本保持一致"""
    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    angle_offsets = np.arange(angle_range[0], angle_range[1] + 1, step)
    perturbed_vs, perturbed_angles, angle_diffs = [], [], []

    for offset in angle_offsets:
        angle_rad = np.radians(offset)
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        angle_diff = angle_between(v, v_main)
        if angle_diff <= theta_max:
            perturbed_vs.append(v)
            perturbed_angles.append(offset)
            angle_diffs.append(angle_diff)

    sorted_indices = np.argsort(angle_diffs)
    top_indices = sorted_indices[:top_k]

    valid_vs = [perturbed_vs[i] for i in top_indices]
    valid_angles = [perturbed_angles[i] for i in top_indices]

    print(f"✅ 生成了 {len(valid_vs)} 个有效扰动方向")
    for i, (v, a) in enumerate(zip(valid_vs, valid_angles)):
        print(f"  扰动{i+1}: angle={a:.1f}°, v=({v[0]:.4f}, {v[1]:.4f})")

    return valid_vs, valid_angles


def run_dpo_generation(
    result_dir,
    valid_vs,
    valid_angles,
    main_v,
    prompt_limit=2000,
    batch_size=32,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
):
    """DPO RPS生成函数 - 模型切换为 Mistral-7B-Instruct-v0.2"""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm.auto import tqdm
    import torch
    import os
    import random
    import numpy as np
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(result_dir, exist_ok=True)

    # 加载数据集（HelpSteer）
    ds = load_dataset("nvidia/HelpSteer", split="validation")
    prompts = ds["prompt"][:prompt_limit]
    prompt_ids = list(range(len(prompts)))

    # 加载模型与tokenizer
    print("🤖 Loading model (mistralai/Mistral-7B-Instruct-v0.2)...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "30GB"} if torch.cuda.is_available() else None,
        ).to(device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tokenizer.padding_side = "left"  # Critical for decoder-only models
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ DPO model loaded!")

    except Exception as e:
        print(f"❌ DPO model loading failed: {e}")
        print("🔄 尝试使用CPU卸载...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            tokenizer.padding_side = "left"  # Critical for decoder-only models
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✅ DPO model loaded on CPU!")
        except Exception as e2:
            print(f"❌ DPO model也加载失败: {e2}")
            return

    def build_input(prompt, v1, v2):
        h = int(np.round(v1 * 100))
        v = int(np.round(v2 * 100))
        sys_instruction = (
            f"You are a helpful assistant. Your response should maximize "
            f"weighted rating = helpfulness*{h} + verbosity*{v}."
        )
        # 与参考脚本一致：将系统提示合并到user消息中
        return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

    def generate_response_batch(prompts_batch, prompt_ids_batch, v1, v2):
        # Clear cache before processing batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        input_ids_list = []
        for prompt in prompts_batch:
            messages = build_input(prompt, v1, v2)
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )[0]
            input_ids_list.append(input_ids.to(device))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)

        attention_mask = (input_ids_padded != tokenizer.pad_token_id).to(device)
        max_new_tokens = 512  # Reduced from 512 to save memory

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids_padded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,  # Enable KV cache for efficiency
            )

        responses = []
        for i, input_ids in enumerate(input_ids_list):
            generated_tokens = outputs[i][input_ids.shape[0]:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_ids_batch[i],
                "prompt": prompts_batch[i],
                "response": decoded,
            })
        return responses

    # 主循环
    for i, (v_vec, angle_deg) in enumerate(zip(valid_vs, valid_angles)):
        v1, v2 = v_vec[0], v_vec[1]
        output_file = os.path.join(result_dir, f"dpo_rps_angle{int(angle_deg)}.csv")
        print(f"\n🚀 Generating for direction {i}: angle ≈ {angle_deg}°, v = ({v1:.4f}, {v2:.4f})")

        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            done_prompt_ids = set(existing_df["prompt_id"].unique())
            results = existing_df.to_dict("records")
            print(f"🔁 Resuming from previous run: {len(done_prompt_ids)} prompts already completed.")
        else:
            done_prompt_ids = set()
            results = []

        for start in tqdm(range(0, len(prompts), batch_size), desc=f"Generating angle {angle_deg}"):
            end = min(start + batch_size, len(prompts))
            batch_prompts_all = prompts[start:end]
            batch_ids_all = prompt_ids[start:end]

            unprocessed_indices = [j for j, pid in enumerate(batch_ids_all) if pid not in done_prompt_ids]
            if not unprocessed_indices:
                continue

            batch_prompts = [batch_prompts_all[j] for j in unprocessed_indices]
            batch_ids = [batch_ids_all[j] for j in unprocessed_indices]

            try:
                batch_outputs = generate_response_batch(batch_prompts, batch_ids, v1, v2)
                for item in batch_outputs:
                    item.update({
                        "v1_p": round(v1, 4),
                        "v2_p": round(v2, 4),
                        "direction_index": i,
                        "valid_angle": round(angle_deg, 1),
                        "main_v1": round(main_v[0], 4),
                        "main_v2": round(main_v[1], 4),
                    })
                    results.append(item)

                pd.DataFrame(batch_outputs).to_csv(
                    output_file, mode='a', index=False,
                    header=not os.path.exists(output_file)
                )

            except Exception as e:
                print(f"⚠️ Error at batch {start}-{end}: {e}")
                # Clear GPU memory after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Continue with smaller batch if OOM error
                if "out of memory" in str(e).lower():
                    print("🔄 Attempting with smaller batch size...")
                    # Try processing items one by one
                    for single_prompt, single_id in zip(batch_prompts, batch_ids):
                        try:
                            single_output = generate_response_batch([single_prompt], [single_id], v1, v2)
                            for item in single_output:
                                item.update({
                                    "v1_p": round(v1, 4),
                                    "v2_p": round(v2, 4),
                                    "direction_index": i,
                                    "valid_angle": round(angle_deg, 1),
                                    "main_v1": round(main_v[0], 4),
                                    "main_v2": round(main_v[1], 4),
                                })
                                results.append(item)
                            
                            pd.DataFrame(single_output).to_csv(
                                output_file, mode='a', index=False,
                                header=not os.path.exists(output_file)
                            )
                        except Exception as single_e:
                            print(f"⚠️ Failed single item {single_id}: {single_e}")
                            continue

        print(f"✅ Final saved {len(results)} responses to {output_file}")


def main():
    """主函数 - 使用HelpSteer prompts与Mistral模型进行生成"""
    output_dir = "/content/drive/MyDrive/sft_rps_helpsteer_outputs"
    os.makedirs(output_dir, exist_ok=True)

    directions_to_test = ["v3", "v4", "v5", "v6","v7","v8","v9","v10"]

    print(f"\n🚀 开始生成DPO RPS响应 (Mistral-7B-Instruct-v0.2)")
    print(f"📐 将测试方向: {directions_to_test}")

    for direction_name in directions_to_test:
        if direction_name not in DPO_PREFERENCE_DIRECTIONS:
            print(f"⚠️ 跳过未定义的方向: {direction_name}")
            continue

        direction_info = DPO_PREFERENCE_DIRECTIONS[direction_name]
        main_v = np.array(direction_info["vector"])

        print(f"\n🎯 处理方向 {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")

        direction_dir = os.path.join(output_dir, direction_name)
        os.makedirs(direction_dir, exist_ok=True)
        print(f"📂 创建方向目录: {direction_dir}")

        valid_vs, valid_angles = get_dpo_angle_perturbations(
            v_main=main_v,
            angle_range=(-40, 40),
            step=5,
            theta_max=30,
            top_k=5,
        )

        run_dpo_generation(
            result_dir=direction_dir,
            valid_vs=valid_vs,
            valid_angles=valid_angles,
            main_v=main_v,
            prompt_limit=2000,
            batch_size=16,  # Reduced from 64 to prevent CUDA OOM
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        )

    print("\n🎉 所有生成实验完成！")
    print(f"📁 生成结果保存在: {output_dir}")
    print(f"📂 目录结构:")
    for direction in directions_to_test:
        direction_path = os.path.join(output_dir, direction)
        if os.path.exists(direction_path):
            files = os.listdir(direction_path)
            print(f"  {direction}/: {len(files)} 个文件")
            for file in files[:3]:
                print(f"    - {file}")


if __name__ == "__main__":
    main()
