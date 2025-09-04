# simpo_response_generation.py
# SimPO模型在UltraFeedback数据集上生成多响应（无评分版本）

import os
import numpy as np
import pandas as pd
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import time
import random

# 🇨🇳 设置国内镜像，解决网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'

print("🌏 已设置Hugging Face国内镜像: https://hf-mirror.com")

# 定义v3-v10的方向向量（基于论文Table）
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
    """设置环境和随机种子"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return device

def load_simpo_model(device):
    """加载SimPO模型"""
    print("🤖 Loading SimPO model from mirror...")
    try:
        # 加载 Princeton NLP 的 SimPO 模型
        model_name = "princeton-nlp/gemma-2-9b-it-SimPO"
        
        simpo_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 如示例所示
            device_map="auto",
            trust_remote_code=True,
            resume_download=True,
            low_cpu_mem_usage=True
        )
        print("✅ SimPO model loaded successfully!")
        
        simpo_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        simpo_tokenizer.padding_side = "left"
        if simpo_tokenizer.pad_token_id is None:
            simpo_tokenizer.pad_token = simpo_tokenizer.eos_token
        print("✅ SimPO tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading SimPO model: {e}")
        print("🔄 Retrying with alternative settings...")
        # 重试机制
        simpo_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=False,
            low_cpu_mem_usage=True
        ).to(device)
        simpo_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if simpo_tokenizer.pad_token_id is None:
            simpo_tokenizer.pad_token = simpo_tokenizer.eos_token
    
    return simpo_model, simpo_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式（按照论文 A.3 节）"""
    # 按照论文附录中的格式构造输入
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

def generate_responses_for_direction(prompt, prompt_id, direction_name, direction_info, 
                                   simpo_model, simpo_tokenizer, device, num_responses=5):
    """为单个prompt在特定方向上生成多个响应（无评分版本）"""
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        messages = build_dpa_input(prompt, v1, v2)
        input_ids = simpo_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        
        max_input_len = input_ids.shape[1]
        max_new_tokens = 512
        
        with torch.no_grad():
            outputs = simpo_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,
                pad_token_id=simpo_tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=[
                    simpo_tokenizer.convert_tokens_to_ids("<end_of_turn>"), 
                    simpo_tokenizer.eos_token_id
                ]
            )
        
        responses = []
        for i in range(num_responses):
            generated_tokens = outputs[i][input_ids.shape[1]:]
            decoded = simpo_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "direction_name": direction_name,
                "direction_vector": f"({v1:.4f}, {v2:.4f})",
                "angle_degrees": angle,
                "response_id": i + 1,
                "response": decoded
            })
        
        return responses
    
    except Exception as e:
        print(f"⚠️ Error generating responses for prompt {prompt_id} direction {direction_name}: {e}")
        return []

def generate_all_directions(
    prompts, 
    prompt_ids,
    simpo_model, 
    simpo_tokenizer, 
    device,
    output_dir,
    batch_size=16,
    num_responses=5
):
    """为所有方向生成响应"""
    
    start_time = time.time()
    all_results = []
    
    # 为每个方向处理
    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        output_file = os.path.join(output_dir, f"simpo_responses_{direction_name}.csv")
        
        # 检查已有结果，支持断点续跑
        done_prompt_ids = set()
        direction_results = []
        
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                done_prompt_ids = set(existing_df["prompt_id"].unique())
                print(f"🔁 Found existing results for {len(done_prompt_ids)} prompts in {direction_name}")
            except Exception as e:
                print(f"⚠️ Error loading existing file for {direction_name}: {e}")
        
        # 计算剩余需要处理的prompts数量
        remaining_prompts = [pid for pid in prompt_ids if pid not in done_prompt_ids]
        print(f" {direction_name}: 已处理 {len(done_prompt_ids)} 个，剩余 {len(remaining_prompts)} 个")
        
        # 批量处理prompts
        for start in tqdm(range(0, len(prompts), batch_size), 
                         desc=f"Processing {direction_name} (剩余{len(remaining_prompts)}个)"):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_ids = prompt_ids[start:end]
            
            # 跳过已处理的prompts
            unprocessed_indices = [i for i, pid in enumerate(batch_ids) if pid not in done_prompt_ids]
            if not unprocessed_indices:
                continue
                
            batch_results = []
            
            for i in unprocessed_indices:
                prompt = batch_prompts[i]
                prompt_id = batch_ids[i]
                
                # 生成多个响应（无评分）
                responses = generate_responses_for_direction(
                    prompt, prompt_id, direction_name, direction_info,
                    simpo_model, simpo_tokenizer, device, num_responses
                )
                
                if responses:
                    batch_results.extend(responses)
            
            # 保存批处理结果
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                
                if not os.path.exists(output_file):
                    df_batch.to_csv(output_file, index=False)
                else:
                    df_batch.to_csv(output_file, mode='a', header=False, index=False)
                
                direction_results.extend(batch_results)
                print(f"✅ Saved batch for {direction_name}, total processed: {len(direction_results)}")
        
        all_results.extend(direction_results)
        print(f"✅ Completed direction {direction_name}: {len(direction_results)} responses")
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed_time:.1f} seconds")
    print(f"📊 Total responses generated: {len(all_results)}")
    
    return all_results

def main():
    """主函数"""
    # 输出目录设置：支持本地和Colab环境
    local_dir = "/mnt/rps_project/data/simpo_outputs"
    colab_dir = "/content/drive/MyDrive/simpo_outputs"
    
    # 检测环境并设置输出目录
    if os.path.exists("/content/drive/MyDrive"):
        result_dir = colab_dir
        print("🔍 Detected Colab environment, using Google Drive output")
    else:
        result_dir = local_dir
        print("🔍 Detected local environment, using local output")
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据集
    print("📦 Loading prompts from UltraFeedback test_prefs via mirror...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        prompts = ds["prompt"][:2000]  # 使用2000个prompts
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Loaded {len(prompts)} prompts successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Retrying dataset loading...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", trust_remote_code=True)
        prompts = ds["prompt"][:2000]
        prompt_ids = list(range(len(prompts)))
    
    # 加载模型
    simpo_model, simpo_tokenizer = load_simpo_model(device)
    
    # 显示将要处理的方向
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    print(f"\n🚀 Starting generation for {len(prompts)} prompts across all directions...")
    print(f"💾 Output directory: {result_dir}")
    
    results = generate_all_directions(
        prompts=prompts,
        prompt_ids=prompt_ids,
        simpo_model=simpo_model,
        simpo_tokenizer=simpo_tokenizer,
        device=device,
        output_dir=result_dir,
        batch_size=16,
        num_responses=5  # 每个prompt生成5个response
    )
    
    print(f"\n✅ All done! Results saved to {result_dir}")
    
    # 显示统计信息
    print(f"📈 Final statistics:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"simpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"  {direction_name}: {len(df)} responses generated")

if __name__ == "__main__":
    main()
