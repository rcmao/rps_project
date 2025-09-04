# Colab版本的SimPO RPS扰动实验 - 简化版（仅生成，不评分）
# 直接在Colab中运行，只生成扰动方向的响应，不进行评分
# 使用princeton-nlp/gemma-2-9b-it-SimPO模型

# 安装必要的包
!pip install transformers datasets torch pandas numpy tqdm accelerate

# 设置内存优化
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import json
import random
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
from collections import Counter
from google.colab import drive

# 挂载Google Drive
drive.mount('/content/drive')

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ 环境设置完成")

# 定义SimPO的主方向向量
SIMPO_PREFERENCE_DIRECTIONS = {
    "v3": {"vector": (0.9848, 0.1736), "angle": 10},
    "v4": {"vector": (0.9659, 0.2588), "angle": 15}, 
    "v5": {"vector": (0.9397, 0.3420), "angle": 20},
    "v6": {"vector": (0.9063, 0.4226), "angle": 25},
    "v7": {"vector": (0.8660, 0.5000), "angle": 30},
    "v8": {"vector": (0.8192, 0.5736), "angle": 35},
    "v9": {"vector": (0.7660, 0.6428), "angle": 40},
    "v10": {"vector": (0.7071, 0.7071), "angle": 45},
}

def get_simpo_angle_perturbations(v_main, angle_range=(-40, 40), step=5, theta_max=30, top_k=5):
    """为SimPO方向生成角度扰动 - 与论文和angle_based.py保持一致"""
    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # 构造单位扰动向量 - 与angle_based.py保持一致
    angle_offsets = np.arange(angle_range[0], angle_range[1] + 1, step)
    perturbed_vs = []
    perturbed_angles = []
    angle_diffs = []

    for offset in angle_offsets:
        angle_rad = np.radians(offset)  # 直接使用offset，不加上main_angle
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        angle_diff = angle_between(v, v_main)
        if angle_diff <= theta_max:
            perturbed_vs.append(v)
            perturbed_angles.append(offset)  # 保存相对角度，与angle_based.py一致
            angle_diffs.append(angle_diff)

    sorted_indices = np.argsort(angle_diffs)
    top_indices = sorted_indices[:top_k]
    
    valid_vs = [perturbed_vs[i] for i in top_indices]
    valid_angles = [perturbed_angles[i] for i in top_indices]
    
    print(f"✅ 生成了 {len(valid_vs)} 个有效扰动方向")
    for i, (v, a) in enumerate(zip(valid_vs, valid_angles)):
        print(f"  扰动{i+1}: angle={a:.1f}°, v=({v[0]:.4f}, {v[1]:.4f})")
    
    return valid_vs, valid_angles

def run_simpo_generation(
    result_dir,
    valid_vs,
    valid_angles,
    main_v,
    prompt_limit=2000,  # 完整数据集
    batch_size=4,       # 减小batch size
    model_name="princeton-nlp/gemma-2-9b-it-SimPO"
):
    """SimPO RPS生成函数 - Colab优化版本"""
    
    # === 设置设备与种子 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(result_dir, exist_ok=True)

    # === 加载数据集 ===
    print("📊 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:prompt_limit]
    prompt_ids = list(range(prompt_limit))
    print(f"✅ Loaded {len(prompts)} prompts")

    # === 加载SimPO模型与 tokenizer ===
    print(f"🤖 Loading SimPO model: {model_name}...")
    try:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Colab优化的模型加载配置
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "12GB"} if torch.cuda.is_available() else None  # Colab GPU限制
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ SimPO model loaded successfully!")
        
    except Exception as e:
        print(f"❌ SimPO model loading failed: {e}")
        print("🔄 尝试使用8bit量化...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✅ SimPO model loaded with 8bit quantization!")
            
        except Exception as e2:
            print(f"❌ SimPO model也加载失败: {e2}")
            return
    
    def build_input(prompt, v1, v2):
        """构建输入 - 与DPO版本保持一致"""
        h = int(np.round(v1 * 100))
        v = int(np.round(v2 * 100))
        sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
        return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

    def generate_response_batch(prompts_batch, prompt_ids_batch, v1, v2):
        """批量生成响应 - Colab优化版本"""
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
        max_input_len = input_ids_padded.shape[1]
        max_new_tokens = min(512, 1024 - max_input_len)  # 调整token数量为512
        
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
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                    tokenizer.eos_token_id
                ] if hasattr(tokenizer, 'convert_tokens_to_ids') else tokenizer.eos_token_id
            )
        
        responses = []
        for i, input_ids in enumerate(input_ids_list):
            generated_tokens = outputs[i][input_ids.shape[0]:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_ids_batch[i],
                "prompt": prompts_batch[i],
                "response": decoded
            })
        return responses
    
    # === 主循环 ===
    for i, (v_vec, angle_deg) in enumerate(zip(valid_vs, valid_angles)):
        v1, v2 = v_vec[0], v_vec[1]
        output_file = os.path.join(result_dir, f"simpo_rps_angle{int(angle_deg)}.csv")
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
                        "model_type": "SimPO"
                    })
                    results.append(item)

                # 保存批次结果
                pd.DataFrame(batch_outputs).to_csv(
                    output_file, mode='a', index=False,
                    header=not os.path.exists(output_file)
                )

                # Colab内存管理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Error at batch {start}-{end}: {e}")
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"✅ Final saved {len(results)} responses to {output_file}")

def main():
    """主函数 - 使用SimPO模型运行RPS扰动实验（Colab版本）"""
    # 设置输出目录
    output_dir = "/content/drive/MyDrive/simpo_rps_generated"
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个方向运行生成实验 - 测试所有方向
    directions_to_test = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    
    print(f"\n🚀 开始生成SimPO RPS响应 (完整版本)")
    print(f"📐 将测试方向: {directions_to_test}")
    print(f"📊 每个方向测试2000个prompts")
    print(f"🤖 使用模型: princeton-nlp/gemma-2-9b-it-SimPO")
    
    for direction_name in directions_to_test:
        if direction_name not in SIMPO_PREFERENCE_DIRECTIONS:
            print(f"⚠️ 跳过未定义的方向: {direction_name}")
            continue
            
        direction_info = SIMPO_PREFERENCE_DIRECTIONS[direction_name]
        main_v = np.array(direction_info["vector"])
        
        print(f"\n🎯 处理方向 {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        # 为每个主方向创建子目录
        direction_dir = os.path.join(output_dir, direction_name)
        os.makedirs(direction_dir, exist_ok=True)
        print(f"📂 创建方向目录: {direction_dir}")
        
        # 生成扰动方向
        valid_vs, valid_angles = get_simpo_angle_perturbations(
            v_main=main_v,
            angle_range=(-40, 40),  # 与论文一致
            step=5,
            theta_max=30,
            top_k=5
        )
        
        # 运行SimPO生成
        run_simpo_generation(
            result_dir=direction_dir,
            valid_vs=valid_vs,
            valid_angles=valid_angles,
            main_v=main_v,
            prompt_limit=2000,  # 完整数据集
            batch_size=2,       # 进一步减小batch size
            model_name="princeton-nlp/gemma-2-9b-it-SimPO"
        )
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(3)  # 给GPU内存清理更多时间
    
    print("\n🎉 所有SimPO生成实验完成！")
    print(f"📁 生成结果保存在: {output_dir}")
    print(f"📂 目录结构:")
    for direction in directions_to_test:
        direction_path = os.path.join(output_dir, direction)
        if os.path.exists(direction_path):
            files = os.listdir(direction_path)
            print(f"  {direction}/: {len(files)} 个文件")
            for file in files[:3]:  # 显示前3个文件
                print(f"    - {file}")

# 运行主函数
if __name__ == "__main__":
    main()
