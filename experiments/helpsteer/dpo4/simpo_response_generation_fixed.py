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
import re

# 🔄 服务器环境设置
print("🚀 Running FIXED version with enhanced vLLM error handling")
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 🇨🇳 设置国内镜像，解决网络访问问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 40:
            print("🔧 High-end GPU detected, using optimized settings")
        elif gpu_memory >= 24:
            print("🔧 Mid-range GPU detected, using balanced settings")
        else:
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
    """根据GPU内存获取最优设置 - 针对Mistral模型优化"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 40:
            return {
                "batch_size": 2,  # 🔧 减小批处理大小避免编码问题
                "max_new_tokens": 256,  # 🔧 减少token数量
                "torch_dtype": torch.bfloat16,
                "device_map": "auto"
            }
        elif gpu_memory >= 24:
            return {
                "batch_size": 1,  # 🔧 更小的批处理
                "max_new_tokens": 256,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
        else:
            return {
                "batch_size": 1,  # 🔧 单个处理
                "max_new_tokens": 128,
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
    else:
        return {
            "batch_size": 1,
            "max_new_tokens": 64,
            "torch_dtype": torch.float32,
            "device_map": None
        }

def clean_text_for_vllm(text):
    """清理文本中可能导致vLLM编码问题的特殊字符"""
    if not isinstance(text, str):
        return str(text)
    
    # 移除或替换可能导致问题的特殊字符
    # 移除emoji和特殊Unicode字符
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # 表情符号
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # 符号和象形文字
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # 交通和地图符号
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # 国旗
    text = re.sub(r'[\U00002600-\U000026FF]', '', text)  # 杂项符号
    text = re.sub(r'[\U00002700-\U000027BF]', '', text)  # 装饰符号
    
    # 替换其他可能有问题的字符
    text = text.replace('🍿', 'popcorn')  # 特别处理日志中出现的emoji
    text = text.replace('🎯', 'target')
    text = text.replace('✅', 'checkmark')
    text = text.replace('❌', 'x')
    text = text.replace('⚠️', 'warning')
    
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def deduplicate_prompts(dataset, prompt_field="prompt"):
    """基于prompt字段对数据集进行去重"""
    print(f"🔄 开始基于'{prompt_field}'字段进行去重...")
    original_count = len(dataset)
    
    unique_data = {}
    seen_prompts = set()
    
    for idx, example in enumerate(dataset):
        prompt = example[prompt_field]
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            unique_data[idx] = example
    
    unique_dataset = list(unique_data.values())
    unique_count = len(unique_dataset)
    
    print(f"✅ 去重完成: {original_count} -> {unique_count} (去除了 {original_count - unique_count} 个重复项)")
    
    return unique_dataset

def load_simpo_model(device, model_name):
    """加载SimPO模型(vLLM) - 增强版本，专门处理Mistral模型"""
    settings = get_optimal_settings()
    print(f"🔧 Using enhanced settings for Mistral compatibility: {settings}")
    
    print(f"🤖 Loading model {model_name} with enhanced vLLM settings...")
    try:
        # 🔧 针对Mistral模型的特殊vLLM配置
        simpo_model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,  # 降低到30%
            max_model_len=2048,          # 🔧 减少最大序列长度
            trust_remote_code=True,
            # 🔧 添加Mistral专用参数
            enforce_eager=True,          # 强制使用eager模式，避免编译问题
            disable_custom_all_reduce=True,  # 禁用自定义reduce操作
        )
        print(f"✅ Model {model_name} loaded successfully with enhanced vLLM!")
        
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
        
        # 🔧 特殊处理Mistral tokenizer
        if "mistral" in model_name.lower():
            print("🔧 Applying Mistral-specific tokenizer settings")
            # 确保tokenizer正确处理特殊token
            if hasattr(simpo_tokenizer, 'add_bos_token'):
                simpo_tokenizer.add_bos_token = True
            if hasattr(simpo_tokenizer, 'add_eos_token'):
                simpo_tokenizer.add_eos_token = False
        
        print("✅ Enhanced tokenizer loaded successfully!")
        
        if simpo_tokenizer.chat_template is None:
            print("🔧 Setting chat template for model")
        else:
            print("✅ Model already has chat template")
        
    except Exception as e:
        print(f"❌ Error loading model {model_name} with enhanced vLLM: {e}")
        return None, None
    
    print("✅ Enhanced vLLM model is ready.")
    
    return simpo_model, simpo_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA模型的输入格式 - 增强版本，清理特殊字符"""
    # 🔧 清理prompt中的特殊字符
    clean_prompt = clean_text_for_vllm(prompt)
    
    angle_rad = np.arctan2(v2, v1)
    weight_helpfulness = int(np.round(np.cos(angle_rad) * 100))
    weight_verbosity = int(np.round(np.sin(angle_rad) * 100))
    
    # 🔧 简化系统指令，避免特殊字符
    sys_instruction = f"You are a helpful assistant. Maximize weighted rating = helpfulness*{weight_helpfulness} + verbosity*{weight_verbosity}"
    
    return [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": clean_prompt}
    ]

def generate_responses_for_direction_vllm_enhanced(prompts_batch, prompt_ids_batch, direction_name, direction_info, 
                                                 simpo_model, simpo_tokenizer, num_responses=5, max_retries=3):
    """使用增强的vLLM为一批prompts生成响应，包含重试机制"""
    
    for attempt in range(max_retries):
        try:
            v1, v2 = direction_info["vector"]
            angle = direction_info["angle"]
            
            settings = get_optimal_settings()
            
            # 🔧 为每个prompt单独处理，避免批处理编码问题
            all_results = []
            
            for i, prompt in enumerate(prompts_batch):
                prompt_id = prompt_ids_batch[i]
                
                try:
                    # 构建输入
                    input_data = build_dpa_input(prompt, v1, v2)
                    
                    # 🔧 清理chat template输出
                    input_text = simpo_tokenizer.apply_chat_template(
                        input_data, add_generation_prompt=True, tokenize=False
                    )
                    input_text = clean_text_for_vllm(input_text)
                    
                    # 🔧 更保守的采样参数
                    sampling_params = SamplingParams(
                        n=num_responses,
                        temperature=0.6,  # 降低temperature
                        top_p=0.8,        # 降低top_p
                        repetition_penalty=1.05,  # 降低repetition_penalty
                        max_tokens=settings["max_new_tokens"],
                        stop=["\n\n\n", "<|endoftext|>", "</s>"],  # 添加停止符
                    )
                    
                    # 单独生成
                    outputs = simpo_model.generate([input_text], sampling_params)
                    
                    # 处理结果
                    for output in outputs:
                        for resp_idx, candidate in enumerate(output.outputs):
                            response_text = clean_text_for_vllm(candidate.text)
                            resp_data = {
                                "prompt_id": prompt_id,
                                "prompt": clean_text_for_vllm(prompt),
                                "direction_name": direction_name,
                                "direction_vector": f"({v1:.4f}, {v2:.4f})",
                                "angle_degrees": angle,
                                "response_id": resp_idx + 1,
                                "response": response_text
                            }
                            all_results.append(resp_data)
                
                except Exception as prompt_error:
                    print(f"⚠️ Error processing prompt {prompt_id} in direction {direction_name}: {prompt_error}")
                    # 继续处理下一个prompt
                    continue
            
            return all_results
        
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for direction {direction_name}: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"❌ All attempts failed for direction {direction_name}")
                return []
    
    return []

def generate_all_directions(
    prompts, 
    prompt_ids,
    simpo_model, 
    simpo_tokenizer, 
    device,
    output_dir,
    batch_size=2,  # 🔧 默认更小的批处理
    num_responses=5
):
    """为所有方向生成响应 - 增强版本"""
    
    start_time = time.time()
    total_responses_generated = 0
    
    settings = get_optimal_settings()
    
    # 🔧 进一步减小批处理大小，特别是对于问题方向
    problematic_directions = {"v9", "v10"}
    
    for direction_name, direction_info in PREFERENCE_DIRECTIONS.items():
        print(f"\n🎯 Processing direction {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
        
        # 🔧 对问题方向使用更小的批处理
        current_batch_size = 1 if direction_name in problematic_directions else batch_size
        print(f"🔧 Using batch_size: {current_batch_size} for direction {direction_name}")
        
        output_file = os.path.join(output_dir, f"simpo_responses_{direction_name}.csv")
        
        # 检查已有结果
        done_prompt_ids = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                valid_responses = existing_df[existing_df["response"].astype(str).str.strip() != ""]
                response_counts = valid_responses["prompt_id"].value_counts()
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
        for start in tqdm(range(0, len(unprocessed_prompts), current_batch_size), 
                         desc=f"Processing {direction_name}"):
            end = min(start + current_batch_size, len(unprocessed_prompts))
            batch_prompts = unprocessed_prompts[start:end]
            batch_ids = unprocessed_pids[start:end]
            
            # 使用增强的vLLM生成
            batch_results = generate_responses_for_direction_vllm_enhanced(
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
                
                # 🔧 添加小延迟，避免GPU过载
                if direction_name in problematic_directions:
                    time.sleep(0.1)
        
        total_responses_generated += len(direction_results)
        print(f"✅ Completed direction {direction_name}: {len(direction_results)} responses")
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 All directions completed in {elapsed_time:.1f} seconds")
    print(f"📊 Total responses generated: {total_responses_generated}")
    
    return

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Enhanced SimPO Response Generation with vLLM using HelpSteer2 dataset")
    
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
        help="Which split of HelpSteer2 dataset to use"
    )
    
    parser.add_argument(
        "--enable_deduplication", 
        action="store_true",
        default=True,
        help="Enable prompt deduplication (default: True)"
    )
    
    return parser.parse_args()

def main():
    """主函数 - 增强版本"""
    args = parse_args()
    
    model_short_name = args.model_name.split("/")[-1]
    result_dir = os.path.join(args.output_dir, model_short_name)
    os.makedirs(result_dir, exist_ok=True)
    print(f"📁 Output directory: {result_dir}")
    print(f"🤖 Using ENHANCED model: {args.model_name}")
    print(f"📊 Will process up to {args.num_prompts} prompts (after deduplication)")
    
    device = setup_environment()
    
    # 加载HelpSteer2数据集
    print(f"📦 Loading prompts from HelpSteer2 {args.dataset_split} split via mirror...")
    try:
        ds = load_dataset("nvidia/HelpSteer2", split=args.dataset_split)
        print(f"✅ Loaded HelpSteer2 dataset with {len(ds)} examples!")
        
        raw_prompts = ds["prompt"]
        print(f"📊 Original dataset size: {len(raw_prompts)}")
        
        if args.enable_deduplication:
            dataset_for_dedup = [{"prompt": prompt} for prompt in raw_prompts]
            unique_dataset = deduplicate_prompts(dataset_for_dedup, "prompt")
            prompts = [item["prompt"] for item in unique_dataset]
        else:
            print("⚠️ Deduplication disabled, using all prompts")
            prompts = raw_prompts
        
        if len(prompts) > args.num_prompts:
            prompts = prompts[:args.num_prompts]
            print(f"📊 Limited to {args.num_prompts} prompts")
        
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Final dataset size: {len(prompts)} prompts")
        
    except Exception as e:
        print(f"❌ Error loading HelpSteer2 dataset: {e}")
        print("🔄 Retrying dataset loading with trust_remote_code=True...")
        try:
            ds = load_dataset("nvidia/HelpSteer2", split=args.dataset_split, trust_remote_code=True)
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
            print(f"❌ Failed to load HelpSteer2 dataset: {e2}")
            return
    
    # 加载增强模型
    simpo_model, simpo_tokenizer = load_simpo_model(device, args.model_name)
    
    if simpo_model is None:
        print("❌ Enhanced model loading failed. Exiting.")
        return
    
    print(f"\n📐 Will process {len(PREFERENCE_DIRECTIONS)} directions with ENHANCED error handling:")
    for name, info in PREFERENCE_DIRECTIONS.items():
        print(f"  {name}: {info['vector']} ({info['angle']}°)")
    
    print(f"\n🚀 Starting ENHANCED generation for {len(prompts)} prompts across all directions...")
    print(f"💾 Output directory: {result_dir}")
    
    if args.batch_size is None:
        settings = get_optimal_settings()
        batch_size = settings["batch_size"]
    else:
        batch_size = args.batch_size
    
    print(f"📊 Using enhanced batch_size: {batch_size}")
    
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
    
    print(f"\n✅ All done with ENHANCED processing! Results saved to {result_dir}")
    
    # 显示结果文件
    print(f"\n📁 Results files:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"simpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024
            print(f"  📄 {direction_name}: {file_size:.1f} KB")
    
    # 显示统计信息
    print(f"\n📈 Final statistics:")
    for direction_name in PREFERENCE_DIRECTIONS.keys():
        output_file = os.path.join(result_dir, f"simpo_responses_{direction_name}.csv")
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"  {direction_name}: {len(df)} responses generated")
    
    print(f"\n💡 ENHANCED results are saved to: {result_dir}")
    print(f"💡 This version includes special handling for vLLM encoding issues")

if __name__ == "__main__":
    main()