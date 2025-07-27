# dpo_baseline_generation_optimized.py
# 优化版本的DPO模型响应生成脚本

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
from concurrent.futures import ThreadPoolExecutor
import gc
from datetime import datetime, timedelta

# 🇨🇳 设置国内镜像，解决网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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

def load_models_optimized(device):
    """优化的模型加载，减少内存占用"""
    cache_dir = '/root/autodl-tmp/hf_cache'
    print(f"🤖 Loading DPO model from mirror... (cache: {cache_dir})")
    
    try:
        # 🔧 优化模型加载，使用更激进的量化
        dpo_model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-zephyr-3b",
            torch_dtype=torch.float16,
            load_in_4bit=True,  # 🔧 使用4bit量化进一步减少内存
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            use_cache=True,  # 🔧 启用KV缓存
        )
        print("✅ DPO model loaded successfully with 4bit quantization!")
        
        dpo_tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-zephyr-3b",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        dpo_tokenizer.padding_side = "left"
        
        if dpo_tokenizer.pad_token_id is None:
            if hasattr(dpo_tokenizer, 'unk_token') and dpo_tokenizer.unk_token is not None:
                dpo_tokenizer.pad_token = dpo_tokenizer.unk_token
            else:
                dpo_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                dpo_model.resize_token_embeddings(len(dpo_tokenizer))
        print("✅ DPO tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading DPO model: {e}")
        raise e
    
    # 🔧 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("🏆 Loading Reward model from mirror...")
    try:
        # 🔧 Reward模型强制放在CPU上，减少GPU内存压力
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            device_map="cpu",  # 🔧 强制CPU
            low_cpu_mem_usage=True,
        )
        print("✅ Reward model loaded on CPU!")
        
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("✅ Reward tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading Reward model: {e}")
        raise e
    
    return dpo_model, dpo_tokenizer, reward_model, reward_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式"""
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

def generate_responses_batch_optimized(prompts, prompt_ids, direction_name, direction_info, 
                                     dpo_model, dpo_tokenizer, device, num_responses=3, max_batch_size=4):
    """🔧 优化的批量生成，真正使用批处理"""
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        all_responses = []
        
        # 🔧 分批处理，避免显存溢出
        for batch_start in range(0, len(prompts), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_ids = prompt_ids[batch_start:batch_end]
            
            print(f"    🔄 Processing micro-batch {batch_start//max_batch_size + 1}, prompts {batch_start}-{batch_end-1} ({len(batch_prompts)} prompts)")
            
            # 🔧 为每个response生成单独处理（因为不同的随机性）
            for resp_idx in range(num_responses):
                print(f"      🎯 Generating response {resp_idx + 1}/{num_responses} for {len(batch_prompts)} prompts")
                
                # 批量构建输入
                batch_messages = [build_dpa_input(prompt, v1, v2) for prompt in batch_prompts]
                batch_inputs = []
                
                for messages in batch_messages:
                    tokenized = dpo_tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt",
                        padding=True, return_attention_mask=True, truncation=True
                    )
                    batch_inputs.append(tokenized)
                
                # 🔧 真正的批处理：将所有输入合并
                if batch_inputs:
                    # 合并input_ids和attention_mask
                    input_ids_list = [inp['input_ids'] if isinstance(inp, dict) else inp for inp in batch_inputs]
                    attention_masks = [inp.get('attention_mask') if isinstance(inp, dict) else None for inp in batch_inputs]
                    
                    # Pad到相同长度
                    max_len = max(ids.shape[1] for ids in input_ids_list)
                    padded_input_ids = []
                    padded_attention_masks = []
                    
                    for i, ids in enumerate(input_ids_list):
                        pad_len = max_len - ids.shape[1]
                        padded_ids = torch.nn.functional.pad(ids, (pad_len, 0), value=dpo_tokenizer.pad_token_id)
                        padded_input_ids.append(padded_ids)
                        
                        if attention_masks[i] is not None:
                            padded_mask = torch.nn.functional.pad(attention_masks[i], (pad_len, 0), value=0)
                        else:
                            padded_mask = torch.ones_like(padded_ids)
                        padded_attention_masks.append(padded_mask)
                    
                    # 合并为batch
                    batch_input_ids = torch.cat(padded_input_ids, dim=0).to(device)
                    batch_attention_mask = torch.cat(padded_attention_masks, dim=0).to(device)
                    
                    # 🔧 批量生成
                    with torch.no_grad():
                        batch_outputs = dpo_model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            max_new_tokens=2048,  # 🔧 与论文实验保持一致
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=dpo_tokenizer.pad_token_id,
                            eos_token_id=dpo_tokenizer.eos_token_id,
                            use_cache=True,  # 🔧 使用KV缓存
                        )
                    
                    # 解码结果
                    for i, output in enumerate(batch_outputs):
                        original_len = batch_input_ids[i].shape[0]
                        generated_tokens = output[original_len:]
                        decoded = dpo_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        all_responses.append({
                            "prompt_id": batch_ids[i],
                            "prompt": batch_prompts[i],
                            "direction_name": direction_name,
                            "direction_vector": f"({v1:.4f}, {v2:.4f})",
                            "angle_degrees": angle,
                            "response_id": resp_idx + 1,
                            "response": decoded
                        })
                    
                    # 🔧 清理显存
                    del batch_outputs, batch_input_ids, batch_attention_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return all_responses
    
    except Exception as e:
        print(f"⚠️ Error in batch generation: {e}")
        return []

def score_responses_batch(responses, reward_model, reward_tokenizer):
    """🔧 批量评分响应"""
    try:
        scored_responses = []
        
        # 🔧 批量评分，减少模型调用次数
        batch_size = 8  # reward model批处理大小
        
        for i in range(0, len(responses), batch_size):
            batch_responses = responses[i:i + batch_size]
            
            # 准备批量输入
            batch_texts = []
            for resp in batch_responses:
                template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
                batch_texts.append(template.format(prompt=resp["prompt"], response=resp["response"]))
            
            # 批量tokenize
            inputs = reward_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # 移动到reward model的设备
            model_device = next(reward_model.parameters()).device
            inputs = inputs.to(model_device)
            
            # 批量推理
            with torch.no_grad():
                logits = reward_model(**inputs).logits.cpu().numpy()
            
            # 处理结果
            for j, resp in enumerate(batch_responses):
                resp_logits = logits[j]
                helpfulness = resp_logits[9]  # r_help(x, y)
                verbosity = resp_logits[4]    # r_verb(x, y)
                
                # 获取direction向量
                direction_vector = resp["direction_vector"]
                v1_str, v2_str = direction_vector.strip("()").split(", ")
                v1, v2 = float(v1_str), float(v2_str)
                
                # DPA奖励函数
                dpa_score = v1 * helpfulness + v2 * verbosity
                
                resp.update({
                    "helpfulness": float(helpfulness),
                    "verbosity": float(verbosity),
                    "v1": float(v1),
                    "v2": float(v2),
                    "dpa_score": float(dpa_score)
                })
                
                scored_responses.append(resp)
        
        return scored_responses
    
    except Exception as e:
        print(f"⚠️ Error in batch scoring: {e}")
        return responses  # 返回未评分的responses

def select_best_responses(scored_responses):
    """🔧 选择每个prompt的最佳响应"""
    # 按prompt_id分组
    from collections import defaultdict
    prompt_groups = defaultdict(list)
    
    for resp in scored_responses:
        prompt_groups[resp["prompt_id"]].append(resp)
    
    best_responses = []
    for prompt_id, responses in prompt_groups.items():
        if responses:
            # 选择DPA得分最高的
            best_response = max(responses, key=lambda x: x["dpa_score"])
            best_response["selected_as_best"] = True
            best_response["all_dpa_scores"] = [r["dpa_score"] for r in responses]
            best_response["num_candidates"] = len(responses)
            best_responses.append(best_response)
    
    return best_responses

# 修复后的进度条逻辑
def process_direction_optimized(direction_name, direction_info, prompts, prompt_ids, 
                              dpo_model, dpo_tokenizer, reward_model, reward_tokenizer, 
                              device, output_dir, batch_size=8, num_responses=3):
    """🔧 优化的单方向处理"""
    print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
    
    output_file = os.path.join(output_dir, f"dpo_responses_{direction_name}.csv")
    
    # 检查已有结果
    done_prompt_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            done_prompt_ids = set(existing_df["prompt_id"].unique())
            print(f"🔁 Found existing results for {len(done_prompt_ids)} prompts in {direction_name}")
        except Exception as e:
            print(f"⚠️ Error loading existing file for {direction_name}: {e}")
    
    # 计算剩余prompts
    remaining_indices = [i for i, pid in enumerate(prompt_ids) if pid not in done_prompt_ids]
    remaining_prompts = [prompts[i] for i in remaining_indices]
    remaining_ids = [prompt_ids[i] for i in remaining_indices]
    
    if not remaining_prompts:
        print(f"✅ {direction_name} already complete!")
        return
    
    print(f"📊 {direction_name}: 已处理 {len(done_prompt_ids)} 个，剩余 {len(remaining_prompts)} 个")
    
    # 🔧 分批处理
    all_direction_results = []
    total_batches = (len(remaining_prompts) + batch_size - 1) // batch_size
    direction_start_time = time.time()
    
    # 🕒 正确的进度条创建 - 完全修复版本
    pbar = tqdm(total=total_batches, 
                desc=f"📊 {direction_name}",
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    
    # 🔧 手动控制批处理循环
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining_prompts))
        batch_prompts = remaining_prompts[start:end]
        batch_ids = remaining_ids[start:end]
        
        # 🔧 批量生成响应
        responses = generate_responses_batch_optimized(
            batch_prompts, batch_ids, direction_name, direction_info,
            dpo_model, dpo_tokenizer, device, num_responses, max_batch_size=2
        )
        
        if not responses:
            pbar.update(1)  # 🔧 即使失败也要更新进度条
            continue
        
        # 🔧 批量评分
        scored_responses = score_responses_batch(responses, reward_model, reward_tokenizer)
        
        # 🔧 选择最佳响应
        best_responses = select_best_responses(scored_responses)
        
        # 保存批处理结果
        if best_responses:
            df_batch = pd.DataFrame(best_responses)
            
            if not os.path.exists(output_file):
                df_batch.to_csv(output_file, index=False)
            else:
                df_batch.to_csv(output_file, mode='a', header=False, index=False)
            
            all_direction_results.extend(best_responses)
            progress_prompts = len(done_prompt_ids) + len(all_direction_results)
            
            # 🕒 计算时间统计
            elapsed_time = time.time() - direction_start_time
            avg_time_per_batch = elapsed_time / (batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            estimated_remaining = avg_time_per_batch * remaining_batches
            
            # 🔧 正确更新进度条
            pbar.set_postfix({
                'prompts': f"{progress_prompts}/2000",
                'avg_time': f"{avg_time_per_batch:.1f}s/batch",
                'eta': f"{timedelta(seconds=int(estimated_remaining))}" if estimated_remaining > 0 else "0:00:00"
            })
        
        # 🔧 手动更新进度条
        pbar.update(1)
        
        # 🔧 强制垃圾回收
        del responses, scored_responses, best_responses
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 关闭进度条
    pbar.close()
    
    print(f"✅ Completed direction {direction_name}: {len(all_direction_results)} responses")
    return all_direction_results

def main():
    """主函数"""
    # 设置输出目录
    result_dir = "/root/rps/data/dpo_outputs"
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查已有数据
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
        print(f"\n✅ Found existing data in {result_dir}")
        print(f"📊 Summary:")
        total_responses = 0
        for direction_name, count in existing_files:
            print(f"  {direction_name}: {count} responses")
            total_responses += count
        print(f"  Total: {total_responses} responses across {len(existing_files)} directions")
        
        user_input = input("\n🤔 Data already exists. Continue anyway? (y/N): ").strip().lower()
        if user_input != 'y':
            print("🛑 Stopping execution.")
            return
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据集
    print("📦 Loading prompts from UltraFeedback...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        prompts = ds["prompt"][:2000]  # 🔧 可以调整数量进行测试
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Loaded {len(prompts)} prompts successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 🔧 优化的模型加载
    dpo_model, dpo_tokenizer, reward_model, reward_tokenizer = load_models_optimized(device)
    
    # 显示处理计划
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    # 🔧 按方向串行处理，但内部批量化
    start_time = time.time()
    total_directions = len(PREFERENCE_DIRECTIONS)
    
    print(f"\n🚀 Starting generation for {len(prompts)} prompts across {total_directions} directions...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, (direction_name, direction_info) in enumerate(PREFERENCE_DIRECTIONS.items(), 1):
        direction_start = time.time()
        print(f"\n🌟 【{i}/{total_directions}】 Starting direction {direction_name}")
        
        process_direction_optimized(
            direction_name=direction_name,
            direction_info=direction_info,
            prompts=prompts,
            prompt_ids=prompt_ids,
            dpo_model=dpo_model,
            dpo_tokenizer=dpo_tokenizer,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            device=device,
            output_dir=result_dir,
            batch_size=16,  # 🔧 增大批处理大小
            num_responses=3
        )
        
        # 🕒 计算并显示整体进度
        direction_elapsed = time.time() - direction_start
        total_elapsed = time.time() - start_time
        avg_time_per_direction = total_elapsed / i
        remaining_directions = total_directions - i
        estimated_total_remaining = avg_time_per_direction * remaining_directions
        
        print(f"✅ Direction {direction_name} completed in {timedelta(seconds=int(direction_elapsed))}")
        print(f"📊 Overall progress: {i}/{total_directions} directions ({i/total_directions*100:.1f}%)")
        if remaining_directions > 0:
            print(f"⏱️  Estimated remaining time: {timedelta(seconds=int(estimated_total_remaining))}")
            completion_time = datetime.now() + timedelta(seconds=int(estimated_total_remaining))
            print(f"🎯 Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed_time:.1f} seconds")
    
    # 显示最终统计
    print(f"📈 Final statistics:")
    total_responses = 0
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"dpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"  {direction_name}: {len(df)} responses, avg DPA score: {df['dpa_score'].mean():.2f}")
            total_responses += len(df)
    
    print(f"📊 Total responses: {total_responses}")
    print(f"⚡ Average time per response: {elapsed_time/total_responses:.2f}s")

if __name__ == "__main__":
    main() 