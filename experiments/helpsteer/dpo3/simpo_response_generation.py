import os
import numpy as np
import pandas as pd
import math
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import time
import random
from vllm import LLM, SamplingParams

# 🔄 服务器环境设置
print("🚀 Running in Server environment with vLLM acceleration")
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 🇨🇳 设置国内镜像，解决网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'
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

def deduplicate_prompts(dataset, prompt_field="prompt"):
    """基于prompt字段对数据集进行去重"""
    print(f"🔄 开始基于'{prompt_field}'字段进行去重...")
    original_count = len(dataset)
    
    # 使用字典保存唯一的prompts，保持原始索引
    unique_data = {}
    seen_prompts = set()
    
    for idx, example in enumerate(dataset):
        prompt = example[prompt_field]
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_data[idx] = example
    
    # 转换回列表格式，保持原有数据结构
    unique_dataset = list(unique_data.values())
    unique_count = len(unique_dataset)
    
    print(f"✅ 去重完成: {original_count} -> {unique_count} (去除了 {original_count - unique_count} 个重复项)")
    
    return unique_dataset

def load_simpo_model(device, model_name):
    """加载SimPO模型(vLLM)"""
    # 🔄 获取最优设置
    settings = get_optimal_settings()
    print(f"🔧 Using settings: {settings}")
    
    print(f"🤖 Loading model {model_name} with vLLM from mirror...")
    try:
        # 使用vLLM加载生成模型
        simpo_model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.35,  # 降低显存使用率适应可用显存
            max_model_len=4096,          # 明确设置最大序列长度
            trust_remote_code=True,
        )
        print(f"✅ Model {model_name} loaded successfully with vLLM!")
        
        simpo_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        simpo_tokenizer.padding_side = "left"
        if simpo_tokenizer.pad_token_id is None:
            simpo_tokenizer.pad_token = simpo_tokenizer.eos_token
        if simpo_tokenizer.pad_token_id == simpo_tokenizer.eos_token_id:
            simpo_tokenizer.pad_token = "<pad>"
            simpo_tokenizer.pad_token_id = simpo_tokenizer.convert_tokens_to_ids("<pad>")
        print("✅ Tokenizer loaded successfully!")
        
        # 🔧 为 DPA 模型等缺少 chat template 的模型设置合适的模板
        if simpo_tokenizer.chat_template is None:
            # DPA模型需要支持system和user角色的chat template
            print("🔧 Setting chat template for DPA model with system/user role support")
        else:
            print("✅ Model already has chat template")
        
    except Exception as e:
        print(f"❌ Error loading model {model_name} with vLLM: {e}")
        # vLLM加载失败，不再尝试transformers回退，直接退出
        return None, None
    
    print("✅ vLLM model is managed internally.")
    
    return simpo_model, simpo_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式（按照 RLHFlow/DPA-v1-Mistral-7B 的要求）"""
    # 计算权重 - DPA模型使用角度来计算权重
    # v1 对应 helpfulness 权重，v2 对应 verbosity 权重
    angle_rad = np.arctan2(v2, v1)  # 计算弧度
    
    # 根据角度计算权重（按照DPA模型的要求）
    weight_helpfulness = int(np.round(np.cos(angle_rad) * 100))
    weight_verbosity = int(np.round(np.sin(angle_rad) * 100))
    
    # 使用DPA模型要求的系统提示格式
    sys_instruction = f"You are a helpful, respectful, and honest assistant who always responds to the user in a harmless way. Your response should maximize weighted rating = helpfulness*{weight_helpfulness} + verbosity*{weight_verbosity}"
    
    return [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt}
    ]

def generate_responses_for_direction_vllm(prompts_batch, prompt_ids_batch, direction_name, direction_info, 
                                         simpo_model, simpo_tokenizer, num_responses=5):
    """使用vLLM为一批prompts在特定方向上生成多个响应"""
    try:
        v1, v2 = direction_info["vector"]
        angle = direction_info["angle"]
        
        # 获取最优设置
        settings = get_optimal_settings()
        
        # 为vLLM准备所有输入
        vllm_inputs = []
        for prompt in prompts_batch:
            input_data = build_dpa_input(prompt, v1, v2)
            
            # 使用chat template
            input_text = simpo_tokenizer.apply_chat_template(
                input_data, add_generation_prompt=True, tokenize=False
            )
            vllm_inputs.append(input_text)
        
        # 设置vLLM采样参数
        sampling_params = SamplingParams(
            n=num_responses,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=settings["max_new_tokens"],
        )
        
        # 使用vLLM一次性生成所有响应
        outputs = simpo_model.generate(vllm_inputs, sampling_params)
        
        # 处理生成的响应
        batch_results = []
        for i, output in enumerate(outputs):
            prompt_id = prompt_ids_batch[i]
            prompt = prompts_batch[i]
            
            for resp_idx, candidate in enumerate(output.outputs):
                response_text = candidate.text
                resp_data = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "direction_name": direction_name,
                    "direction_vector": f"({v1:.4f}, {v2:.4f})",
                    "angle_degrees": angle,
                    "response_id": resp_idx + 1,
                    "response": response_text
                }
                batch_results.append(resp_data)
        
        return batch_results
    
    except Exception as e:
        print(f"⚠️ Error generating responses for direction {direction_name}: {e}")
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
    """为所有方向生成响应 (vLLM版本)"""
    
    start_time = time.time()
    total_responses_generated = 0
    
    # 获取最优设置
    settings = get_optimal_settings()
    
    # 为每个方向处理
    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        output_file = os.path.join(output_dir, f"simpo_responses_{direction_name}.csv")
        
        # 检查已有结果，支持断点续跑（方案二：只统计有足够非空响应的prompt_id）
        done_prompt_ids = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                # 筛选出非空响应（去除空字符串和只有空白字符的响应）
                valid_responses = existing_df[existing_df["response"].astype(str).str.strip() != ""]
                # 统计每个 prompt_id 有多少个有效响应
                response_counts = valid_responses["prompt_id"].value_counts()
                # 只有达到目标数量的 prompt_id 才算完成
                done_prompt_ids = set(response_counts[response_counts >= num_responses].index)
                total_existing = len(existing_df["prompt_id"].unique())
                valid_completed = len(done_prompt_ids)
                print(f"🔁 Found {total_existing} prompts with records, {valid_completed} prompts with sufficient valid responses in {direction_name}")
            except Exception as e:
                print(f"⚠️ Error loading existing file for {direction_name}: {e}")
        
        # 筛选需要处理的prompts
        unprocessed_indices = [i for i, pid in enumerate(prompt_ids) if pid not in done_prompt_ids]
        if not unprocessed_indices:
            print(f"✅ Direction {direction_name} is already complete.")
            continue
            
        unprocessed_prompts = [prompts[i] for i in unprocessed_indices]
        unprocessed_pids = [prompt_ids[i] for i in unprocessed_indices]
        print(f"📊 {direction_name}: 已处理 {len(done_prompt_ids)} 个，剩余 {len(unprocessed_prompts)} 个")
        
        # 批量处理prompts
        direction_results = []
        for start in tqdm(range(0, len(unprocessed_prompts), batch_size), 
                         desc=f"Processing {direction_name}"):
            end = min(start + batch_size, len(unprocessed_prompts))
            batch_prompts = unprocessed_prompts[start:end]
            batch_ids = unprocessed_pids[start:end]
            
            # 使用vLLM批量生成
            batch_results = generate_responses_for_direction_vllm(
                batch_prompts, batch_ids, direction_name, direction_info,
                simpo_model, simpo_tokenizer, num_responses
            )
            
            # 保存批处理结果
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                
                if not os.path.exists(output_file):
                    df_batch.to_csv(output_file, index=False)
                else:
                    df_batch.to_csv(output_file, mode='a', header=False, index=False)
                
                direction_results.extend(batch_results)
        
        total_responses_generated += len(direction_results)
        print(f"✅ Completed direction {direction_name}: {len(direction_results)} responses")
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed_time:.1f} seconds")
    print(f"📊 Total responses generated: {total_responses_generated}")
    
    return

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SimPO Response Generation with vLLM using HelpSteer dataset")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="princeton-nlp/gemma-2-9b-it-SimPO",
        help="Model name to use for generation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/shiyl/workspace/steer/results/simpo_outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num_prompts", 
        type=int, 
        default=2000,
        help="Number of prompts to process (after deduplication)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size (if not set, will use optimal settings based on GPU memory)"
    )
    
    parser.add_argument(
        "--num_responses", 
        type=int, 
        default=5,
        help="Number of responses to generate per prompt"
    )
    
    parser.add_argument(
        "--dataset_split", 
        type=str, 
        default="validation",
        choices=["train", "validation"],
        help="Which split of HelpSteer dataset to use"
    )
    
    parser.add_argument(
        "--enable_deduplication", 
        action="store_true",
        default=True,
        help="Enable prompt deduplication (default: True)"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置输出目录
    model_short_name = args.model_name.split("/")[-1]
    result_dir = os.path.join(args.output_dir, model_short_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"📁 Output directory: {result_dir}")
    print(f"🤖 Using model: {args.model_name}")
    print(f"📊 Will process up to {args.num_prompts} prompts (after deduplication)")
    
    # 设置环境
    device = setup_environment()
    
    # 加载HelpSteer数据集
    print(f"📦 Loading prompts from HelpSteer {args.dataset_split} split via mirror...")
    try:
        ds = load_dataset("nvidia/HelpSteer", split=args.dataset_split)
        print(f"✅ Loaded HelpSteer dataset with {len(ds)} examples!")
        
        # 提取prompts
        raw_prompts = ds["prompt"]
        print(f"📊 Original dataset size: {len(raw_prompts)}")
        
        # 去重处理
        if args.enable_deduplication:
            # 转换为适合去重的格式
            dataset_for_dedup = [{"prompt": prompt} for prompt in raw_prompts]
            unique_dataset = deduplicate_prompts(dataset_for_dedup, "prompt")
            prompts = [item["prompt"] for item in unique_dataset]
        else:
            print("⚠️ Deduplication disabled, using all prompts")
            prompts = raw_prompts
        
        # 限制数量
        if len(prompts) > args.num_prompts:
            prompts = prompts[:args.num_prompts]
            print(f"📊 Limited to {args.num_prompts} prompts")
        
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Final dataset size: {len(prompts)} prompts")
        
    except Exception as e:
        print(f"❌ Error loading HelpSteer dataset: {e}")
        print("🔄 Retrying dataset loading with trust_remote_code=True...")
        try:
            ds = load_dataset("nvidia/HelpSteer", split=args.dataset_split, trust_remote_code=True)
            raw_prompts = ds["prompt"]
            
            if args.enable_deduplication:
                dataset_for_dedup = [{"prompt": prompt} for prompt in raw_prompts]
                unique_dataset = deduplicate_prompts(dataset_for_dedup, "prompt")
                prompts = [item["prompt"] for item in unique_dataset]
            else:
                prompts = raw_prompts
            
            if len(prompts) > args.num_prompts:
                prompts = prompts[:args.num_prompts]
            
            prompt_ids = list(range(len(prompts)))
            print(f"✅ Final dataset size: {len(prompts)} prompts")
        except Exception as e2:
            print(f"❌ Failed to load HelpSteer dataset: {e2}")
            return
    
    # 加载模型
    simpo_model, simpo_tokenizer = load_simpo_model(device, args.model_name)
    
    # 如果模型加载失败则退出
    if simpo_model is None:
        print("❌ Model loading failed. Exiting.")
        return
    
    # 显示将要处理的方向
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    print(f"\n🚀 Starting generation for {len(prompts)} prompts across all directions with vLLM...")
    print(f"💾 Output directory: {result_dir}")
    
    # 获取batch_size
    if args.batch_size is None:
        settings = get_optimal_settings()
        batch_size = settings["batch_size"]
    else:
        batch_size = args.batch_size
    
    print(f"📊 Using batch_size: {batch_size}")
    
    generate_all_directions(
        prompts=prompts,
        prompt_ids=prompt_ids,
        simpo_model=simpo_model,
        simpo_tokenizer=simpo_tokenizer,
        device=device,
        output_dir=result_dir,
        batch_size=batch_size,
        num_responses=args.num_responses
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
            print(f"  {direction_name}: {len(df)} responses generated")
    
    # 🔄 服务器使用提示
    print(f"\n💡 Results are saved to: {result_dir}")
    print(f"💡 You can find CSV files for each direction in the output directory")

if __name__ == "__main__":
    main()
