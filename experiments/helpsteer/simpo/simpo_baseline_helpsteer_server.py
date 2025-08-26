# simpo_baseline_helpsteer_server.py - 服务器版本
# SimPO模型在HelpSteer数据集上使用DPA方向生成响应并选择最佳回答

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

# 🔄 服务器环境设置
print("🚀 Running in Server environment")
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    
    # 🔄 服务器GPU内存检测和优化
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
        
        # 根据GPU内存自动调整设置
        if gpu_memory >= 40:  # L40S, A100等
            print("🔧 High-end GPU detected, using optimized settings")
        elif gpu_memory >= 24:  # V100, RTX 3090等
            print("🔧 Mid-range GPU detected, using balanced settings")
        else:  # T4, K80等
            print("🔧 Entry-level GPU detected, using conservative settings")
    
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return device

def get_optimal_settings():
    """根据GPU内存获取最优设置"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 40:  # L40S, A100等
            return {
                "batch_size": 16,
                "max_new_tokens": 512,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto"
            }
        elif gpu_memory >= 24:  # V100, RTX 3090等
            return {
                "batch_size": 8,
                "max_new_tokens": 512,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
        else:  # T4, K80等
            return {
                "batch_size": 4,
                "max_new_tokens": 256,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
    else:
        return {
            "batch_size": 2,
            "max_new_tokens": 128,
            "torch_dtype": torch.float32,
            "device_map": None
        }

def load_models(device):
    """加载SimPO模型和Reward模型"""
    # 🔄 获取最优设置
    settings = get_optimal_settings()
    print(f"🔧 Using settings: {settings}")
    
    print("🤖 Loading SimPO model from mirror...")
    try:
        # 🔄 根据GPU内存优化设置
        simpo_model = AutoModelForCausalLM.from_pretrained(
            "princeton-nlp/gemma-2-9b-it-SimPO",  # ✅ SimPO模型
            torch_dtype=settings["torch_dtype"],  # 🔄 动态数据类型
            device_map=settings["device_map"],  # 🔄 动态设备映射
            trust_remote_code=True,
            resume_download=True,  # 支持断点续传
            low_cpu_mem_usage=True  # 🔄 低CPU内存使用
        )
        print("✅ SimPO model loaded successfully!")
        
        simpo_tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/gemma-2-9b-it-SimPO",  # 🔄 替换为对应的tokenizer
            trust_remote_code=True
        )
        simpo_tokenizer.padding_side = "left"
        # 🔄 修复pad token问题
        if simpo_tokenizer.pad_token_id is None:
            simpo_tokenizer.pad_token = simpo_tokenizer.eos_token
        # 确保pad token和eos token不同
        if simpo_tokenizer.pad_token_id == simpo_tokenizer.eos_token_id:
            simpo_tokenizer.pad_token = "<pad>"
            simpo_tokenizer.pad_token_id = simpo_tokenizer.convert_tokens_to_ids("<pad>")
        print("✅ SimPO tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading SimPO model: {e}")
        print("🔄 Retrying with alternative settings...")
        # 重试机制
        simpo_model = AutoModelForCausalLM.from_pretrained(
            "princeton-nlp/gemma-2-9b-it-SimPO",  # 🔄 替换为SimPO模型
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            resume_download=True,
            local_files_only=False,
            low_cpu_mem_usage=True
        ).to(device)
        simpo_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/gemma-2-9b-it-SimPO")  # 🔄 替换为对应的tokenizer
        # 🔄 修复pad token问题
        if simpo_tokenizer.pad_token_id is None:
            simpo_tokenizer.pad_token = simpo_tokenizer.eos_token
        if simpo_tokenizer.pad_token_id == simpo_tokenizer.eos_token_id:
            simpo_tokenizer.pad_token = "<pad>"
            simpo_tokenizer.pad_token_id = simpo_tokenizer.convert_tokens_to_ids("<pad>")
    
    print("🏆 Loading Reward model from mirror...")
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1", 
            trust_remote_code=True,
            resume_download=True,
            torch_dtype=torch.float16,  # 🔄 使用fp16
            low_cpu_mem_usage=True,  # 🔄 低CPU内存使用
            device_map="auto"  # 🔄 自动设备映射
        )
        print("✅ Reward model loaded successfully!")
        
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True
        )
        # 🔄 修复reward tokenizer的padding token问题
        if reward_tokenizer.pad_token_id is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        print("✅ Reward tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading Reward model: {e}")
        print("🔄 Trying alternative reward model...")
        # 可以使用备选模型
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1", 
            trust_remote_code=True,
            resume_download=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1"
        )
        # 🔄 修复reward tokenizer的padding token问题
        if reward_tokenizer.pad_token_id is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # 🔄 确保所有模型都在正确的设备上
    if hasattr(simpo_model, 'device'):
        print(f"✅ SimPO model is on device: {simpo_model.device}")
    else:
        print("✅ SimPO model loaded with device_map='auto'")
    
    if hasattr(reward_model, 'device'):
        print(f"✅ Reward model is on device: {reward_model.device}")
    else:
        print("✅ Reward model loaded with device_map='auto'")
    
    return simpo_model, simpo_tokenizer, reward_model, reward_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式"""
    # 按照论文附录中的格式构造输入
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

def generate_responses_for_direction(prompt, prompt_id, direction_name, direction_info, 
                                   simpo_model, simpo_tokenizer, device, num_responses=3):
    """为单个prompt在特定方向上生成多个响应"""
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        messages = build_dpa_input(prompt, v1, v2)
        input_ids = simpo_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        
        # 🔄 修复attention mask问题
        attention_mask = torch.ones_like(input_ids).to(device)
        
        max_input_len = input_ids.shape[1]
        # 🔄 使用动态设置
        settings = get_optimal_settings()
        max_new_tokens = settings["max_new_tokens"]
        
        with torch.no_grad():
            outputs = simpo_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # 🔄 添加attention mask
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,
                pad_token_id=simpo_tokenizer.pad_token_id,  # 🔄 使用专门的pad token
                top_p=0.9,  # 🔄 添加top_p参数
                repetition_penalty=1.1  # 🔄 添加repetition_penalty参数
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

def score_response_dpa(prompt, response, reward_model, reward_tokenizer, device, v1, v2):
    """使用DPA框架为响应打分"""
    try:
        template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        # 🔄 修复设备不匹配问题
        inputs = reward_tokenizer(
            template.format(prompt=prompt, response=response), 
            return_tensors="pt",
            padding=False,  # 🔄 单个样本不需要padding
            truncation=True,
            max_length=2048
        )
        
        # 确保所有输入都在正确的设备上
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device) if "attention_mask" in inputs else None
        
        # 🔄 额外的设备检查
        if hasattr(reward_model, 'device'):
            expected_device = reward_model.device
        else:
            expected_device = device
        
        # 确保输入在正确的设备上
        input_ids = input_ids.to(expected_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(expected_device)
        
        # 构造输入字典
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        with torch.no_grad():
            logits = reward_model(**model_inputs).logits.squeeze().cpu().numpy()
        
        # 10维输出，对应HelpSteer和UltraFeedback的属性
        # ['helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence', 'helpsteer-complexity', 'helpsteer-verbosity', 
        #  'ultrafeedback-overall_score', "ultrafeedback-instruction_following", "ultrafeedback-truthfulness", "ultrafeedback-honesty", "ultrafeedback-helpfulness"]
        
        helpfulness = logits[0]  # helpsteer-helpfulness
        verbosity = logits[4]    # helpsteer-verbosity
        
        # DPA奖励函数: R(v, y) = v1 * r_help + v2 * r_verb
        dpa_score = v1 * helpfulness + v2 * verbosity
        
        return {
            "helpfulness": float(helpfulness),  # 直接使用原始评分，与图片格式一致
            "verbosity": float(verbosity),      # 直接使用原始评分，与图片格式一致
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
    simpo_model, 
    simpo_tokenizer, 
    reward_model, 
    reward_tokenizer, 
    device,
    output_dir,
    batch_size=16,  # 🔄 从8改为16，L40S显存充足
    num_responses=3
):
    """为所有方向生成和评估响应"""
    
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
                    simpo_model, simpo_tokenizer, device, num_responses
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
    # 🔄 服务器环境输出目录
    result_dir = "/root/rps/data/simpo_baseline_outputs"
    os.makedirs(result_dir, exist_ok=True)
    print(f"📁 Output directory: {result_dir}")
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据集
    print("📦 Loading prompts from HelpSteer via mirror...")
    try:
        ds = load_dataset("nvidia/HelpSteer", split="validation")
        prompts = ds["prompt"][:2000]  # 🔄 修改为2000个prompts
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Loaded {len(prompts)} prompts successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Retrying dataset loading...")
        ds = load_dataset("nvidia/HelpSteer", split="validation", trust_remote_code=True)
        prompts = ds["prompt"][:2000]  # 🔄 修改为2000个prompts
        prompt_ids = list(range(len(prompts)))
    
    # 加载模型
    simpo_model, simpo_tokenizer, reward_model, reward_tokenizer = load_models(device)
    
    # 显示将要处理的方向
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    # 🔄 获取最优设置
    settings = get_optimal_settings()
    batch_size = settings["batch_size"]
    
    print(f"\n🚀 Starting generation for {len(prompts)} prompts across all directions...")
    print(f"🔧 Using batch size: {batch_size}")
    results = generate_and_evaluate_all_directions(
        prompts=prompts,
        prompt_ids=prompt_ids,
        simpo_model=simpo_model,
        simpo_tokenizer=simpo_tokenizer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        device=device,
        output_dir=result_dir,
        batch_size=batch_size,  # 🔄 使用动态batch size
        num_responses=3
    )
    
    print(f"\n✅ All done! Results saved to {result_dir}")
    
    # 🔄 服务器特定功能：显示结果文件
    print(f"\n📁 Results files:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"simpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024  # KB
            print(f"  📄 {direction_name}: {file_size:.1f} KB")
    
    # 显示统计信息
    print(f"\n📈 Final statistics:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"simpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"  {direction_name}: {len(df)} responses, avg DPA score: {df['dpa_score'].mean():.2f}")
    
    # 🔄 服务器使用提示
    print(f"\n💡 Results are saved to: {result_dir}")
    print(f"💡 You can find CSV files for each direction in the output directory")

if __name__ == "__main__":
    main()
