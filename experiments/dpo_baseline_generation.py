# dpo_response_generator.py
# DPO模型在UltraFeedback数据集上使用DPA方向生成响应并选择最佳回答

import os
import numpy as np
import pandas as pd
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
import time
import random

# 🇨🇳 设置国内镜像，解决网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 🔄 修改缓存目录到数据盘 - 使用真正的数据盘路径
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/hf_cache'

print("🌏 已设置Hugging Face国内镜像: https://hf-mirror.com")
print("💾 模型缓存目录: /root/autodl-tmp/hf_cache (150GB数据盘)")

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

def load_models(device):
    """加载DPO模型和Reward模型"""
    cache_dir = '/root/autodl-tmp/hf_cache'
    print(f"🤖 Loading DPO model from mirror... (cache: {cache_dir})")
    try:
        # 🔄 使用真正的DPO训练模型 - zephyr-7b-beta
        dpo_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # ✅ 真正的DPO模型
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分配到GPU/CPU
            trust_remote_code=True,
            cache_dir=cache_dir,  # 🔄 添加缓存目录
            resume_download=True,  # 支持断点续传
            low_cpu_mem_usage=True  # 🔄 减少CPU内存使用
        )
        print("✅ DPO model loaded successfully!")
        
        dpo_tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # 🔄 替换为对应的tokenizer
            trust_remote_code=True,
            cache_dir=cache_dir  # 🔄 添加缓存目录
        )
        dpo_tokenizer.padding_side = "left"
        # 🔧 修复pad_token设置 - zephyr模型使用eos_token作为pad_token
        if dpo_tokenizer.pad_token_id is None:
            dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
        print("✅ DPO tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading DPO model: {e}")
        print("🔄 Retrying with alternative settings...")
        # 重试机制 - 🔄 确保也使用cache_dir
        dpo_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # 🔄 替换为真正的DPO模型
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,  # 🔄 添加缓存目录
            resume_download=True,
            local_files_only=False,
            low_cpu_mem_usage=True
        )
        dpo_tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # 🔄 替换为对应的tokenizer
            cache_dir=cache_dir  # 🔄 添加缓存目录
        )
        if dpo_tokenizer.pad_token_id is None:
            dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
    
    # 🔄 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🏆 Loading Reward model from mirror...")
    try:
        # 🔄 Reward模型放在GPU上，与DPO模型共享显存
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1", 
            trust_remote_code=True,
            cache_dir=cache_dir,  # 🔄 添加缓存目录
            resume_download=True,
            torch_dtype=torch.float16,  # 🔧 使用fp16与DPO模型一致
            device_map="auto",  # 🔧 自动分配设备
            low_cpu_mem_usage=True
        )
        print("✅ Reward model loaded successfully!")
        
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True,
            cache_dir=cache_dir  # 🔄 添加缓存目录
        )
        print("✅ Reward tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading Reward model: {e}")
        print("🔄 Trying alternative reward model...")
        # 🔄 备选方案：将reward模型放在CPU上
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1", 
            trust_remote_code=True,
            cache_dir=cache_dir,  # 🔄 添加缓存目录
            resume_download=True,
            torch_dtype=torch.float16,
            device_map="auto",  # 🔄 自动分配设备
            low_cpu_mem_usage=True
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            cache_dir=cache_dir  # 🔄 添加缓存目录
        )
        print("✅ Reward model loaded successfully")
    
    return dpo_model, dpo_tokenizer, reward_model, reward_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式"""
    # 按照论文附录中的格式构造输入
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

# 不要使用num_return_sequences，改为循环生成
def generate_responses_for_direction(prompt, prompt_id, direction_name, direction_info, 
                                   dpo_model, dpo_tokenizer, device, num_responses=3):
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        messages = build_dpa_input(prompt, v1, v2)
        tokenized = dpo_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            padding=True, return_attention_mask=True, truncation=True
        )
        input_ids = tokenized['input_ids'].to(device) if isinstance(tokenized, dict) else tokenized.to(device)
        attention_mask = tokenized.get('attention_mask', None) if isinstance(tokenized, dict) else None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            # 如果没有attention_mask，创建一个全1的mask
            attention_mask = torch.ones_like(input_ids).to(device)
        
        responses = []
        
        # 🔧 一次性生成多个响应，更高效
        print(f"    🔄 Generating {num_responses} responses for prompt {prompt_id}...")
        with torch.no_grad():
            outputs = dpo_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,  # 🔧 恢复一次性生成多个响应
                pad_token_id=dpo_tokenizer.pad_token_id,
                eos_token_id=dpo_tokenizer.eos_token_id,
                top_p=0.9,  # 🔧 添加top_p参数
                repetition_penalty=1.1  # 🔧 添加repetition_penalty参数
            )
        
        # 🔧 处理多个输出
        for i in range(num_responses):
            generated_tokens = outputs[i][input_ids.shape[1]:]
            decoded = dpo_tokenizer.decode(generated_tokens, skip_special_tokens=True)
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

def score_response_dpa(prompt, response, reward_model, reward_tokenizer, device, v1, v2):
    """使用DPA框架为响应打分"""
    try:
        template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        inputs = reward_tokenizer(
            template.format(prompt=prompt, response=response), 
            return_tensors="pt"
        )
        
        # 🔄 检查reward模型的设备并相应地移动输入
        model_device = next(reward_model.parameters()).device
        inputs = inputs.to(model_device)
        
        with torch.no_grad():
            logits = reward_model(**inputs).logits.squeeze().cpu().numpy()
        
        helpfulness = logits[9]  # r_help(x, y)
        verbosity = logits[4]    # r_verb(x, y)
        
        # DPA奖励函数: R(v, y) = v1 * r_help + v2 * r_verb
        dpa_score = v1 * helpfulness + v2 * verbosity
        
        return {
            "helpfulness": float(helpfulness),
            "verbosity": float(verbosity),
            "v1": float(v1),
            "v2": float(v2),
            "dpa_score": float(dpa_score)
        }
    
    except Exception as e:
        print(f"⚠️ Error scoring response: {e}")
        return {
            "helpfulness": 0.0,
            "verbosity": 0.0,
            "v1": float(v1),
            "v2": float(v2),
            "dpa_score": 0.0
        }

def generate_and_evaluate_all_directions(
    prompts, 
    prompt_ids,
    dpo_model, 
    dpo_tokenizer, 
    reward_model, 
    reward_tokenizer, 
    device,
    output_dir,
    batch_size=8,  # 🔧 增大batch size提高效率
    num_responses=3
):
    """为所有方向生成和评估响应"""
    
    start_time = time.time()
    all_results = []
    
    # 为每个方向处理
    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        output_file = os.path.join(output_dir, f"dpo_responses_{direction_name}.csv")
        
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
        
        v1, v2 = direction_info["vector"]
        
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
                
                # 生成多个响应
                responses = generate_responses_for_direction(
                    prompt, prompt_id, direction_name, direction_info,
                    dpo_model, dpo_tokenizer, device, num_responses
                )
                
                if not responses:
                    continue
                
                # 为每个响应打分
                scored_responses = []
                for resp_data in responses:
                    scores = score_response_dpa(
                        resp_data["prompt"], 
                        resp_data["response"],
                        reward_model,
                        reward_tokenizer,
                        device,
                        v1, v2
                    )
                    
                    resp_data.update(scores)
                    scored_responses.append(resp_data)
                
                # 选择得分最高的响应
                if scored_responses:
                    best_response = max(scored_responses, key=lambda x: x["dpa_score"])
                    best_response["selected_as_best"] = True
                    
                    # 添加其他响应信息
                    best_response["all_dpa_scores"] = [r["dpa_score"] for r in scored_responses]
                    best_response["num_candidates"] = len(scored_responses)
                    
                    batch_results.append(best_response)
            
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
    # 🔄 使用新的输出目录，使用7B模型
    result_dir = "/root/rps/data/dpo_outputs"
    os.makedirs(result_dir, exist_ok=True)
    
    # 🔄 检查已有数据
    existing_files = []
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"dpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            try:
                df = pd.read_csv(output_file)
                existing_files.append((direction_name, len(df)))
                print(f"📁 Found existing data for {direction_name}: {len(df)} responses")
            except Exception as e:
                print(f"⚠️ Error reading {direction_name}: {e}")
    
    if existing_files:
        print(f"\n✅ Found complete baseline data in {result_dir}")
        print(f"📊 Summary:")
        total_responses = 0
        for direction_name, count in existing_files:
            print(f"  {direction_name}: {count} responses")
            total_responses += count
        print(f"  Total: {total_responses} responses across {len(existing_files)} directions")
        
        user_input = input("\n🤔 Data already exists. Continue anyway? (y/N): ").strip().lower()
        if user_input != 'y':
            print("🛑 Stopping execution. Use existing data for analysis.")
            return
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据集
    print("📦 Loading prompts from UltraFeedback via mirror...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        prompts = ds["prompt"][:2000]  # 🔄 修改为2000个prompts
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Loaded {len(prompts)} prompts successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Retrying dataset loading with different methods...")
        
        # 尝试多种方法
        retry_methods = [
            # 方法1：不使用trust_remote_code
            lambda: load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs"),
            # 方法2：尝试使用cache
            lambda: load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", cache_dir="/root/.cache/huggingface"),
            # 方法3：强制重新下载
            lambda: load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", download_mode="force_redownload"),
        ]
        
        for i, method in enumerate(retry_methods, 1):
            try:
                print(f"🔄 Trying method {i}...")
                ds = method()
                prompts = ds["prompt"][:2000]
                prompt_ids = list(range(len(prompts)))
                print(f"✅ Method {i} succeeded! Loaded {len(prompts)} prompts")
                break
            except Exception as retry_error:
                print(f"❌ Method {i} failed: {retry_error}")
                if i == len(retry_methods):
                    print("🚨 All retry methods failed. Please check your network connection.")
                    raise retry_error
    
    # 加载模型
    dpo_model, dpo_tokenizer, reward_model, reward_tokenizer = load_models(device)
    
    # 显示将要处理的方向
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    print(f"\n🚀 Starting generation for {len(prompts)} prompts across all directions...")
    results = generate_and_evaluate_all_directions(
        prompts=prompts,
        prompt_ids=prompt_ids,
        dpo_model=dpo_model,
        dpo_tokenizer=dpo_tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        device=device,
        output_dir=result_dir,
        batch_size=8,  # 🔧 增大batch size提高效率
        num_responses=3  # 进一步加速
    )
    
    print(f"\n✅ All done! Results saved to {result_dir}")
    
    # 显示统计信息
    print(f"📈 Final statistics:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"dpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"  {direction_name}: {len(df)} responses, avg DPA score: {df['dpa_score'].mean():.2f}")

if __name__ == "__main__":
    main()