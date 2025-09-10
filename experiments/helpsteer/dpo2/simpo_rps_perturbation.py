import os
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

    # 🔄 获取最优设置
    settings = get_optimal_settings()
    print(f"🔧 Using settings: {settings}")
    
    # === 加载SimPO模型与 tokenizer ===
    print(f"🤖 Loading SimPO model with vLLM: {model_name}...")
    try:
        # 使用vLLM加载生成模型
        model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,  # 增加显存分配给vLLM
            max_model_len=4096,          # 明确设置最大序列长度
            trust_remote_code=True,
        )
        print("✅ SimPO model loaded successfully with vLLM!")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.pad_token = "<pad>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        print("✅ SimPO tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading SimPO model with vLLM: {e}")
        # vLLM加载失败，不再尝试transformers回退，直接退出
        return
    
    print("✅ vLLM model is managed internally.")
    
    def build_input(prompt, v1, v2):
        """构建输入 - 兼容有无chat template的模型"""
        h = int(np.round(v1 * 100))
        v = int(np.round(v2 * 100))
        sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
        
        # 检查是否有chat_template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            # 有chat template，返回消息格式
            return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]
        else:
            # 没有chat template，直接返回拼接的文本
            return f"{sys_instruction}\n\n{prompt}"

    def generate_response_batch(prompts_batch, prompt_ids_batch, v1, v2):
        """批量生成响应 - vLLM优化版本"""
        # 为vLLM准备所有输入
        vllm_inputs = []
        for prompt in prompts_batch:
            input_data = build_input(prompt, v1, v2)
            
            # 检查是否需要使用chat template
            if isinstance(input_data, list):
                # 有chat template，使用apply_chat_template
                input_text = tokenizer.apply_chat_template(
                    input_data, add_generation_prompt=True, tokenize=False
                )
            else:
                # 没有chat template，直接使用文本
                input_text = input_data
            vllm_inputs.append(input_text)
        
        # 设置vLLM采样参数
        sampling_params = SamplingParams(
            n=1,  # 每个prompt生成1个响应
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=settings["max_new_tokens"],
        )
        
        # 使用vLLM一次性生成所有响应
        outputs = model.generate(vllm_inputs, sampling_params)
        
        responses = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text  # 取第一个响应
            responses.append({
                "prompt_id": prompt_ids_batch[i],
                "prompt": prompts_batch[i],
                "response": response_text
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

def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="SimPO RPS Perturbation with vLLM")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="princeton-nlp/gemma-2-9b-it-SimPO",
        help="Model name to use for generation"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/home/shiyl/workspace/steer/results/simpo_rps_outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--num_prompts", 
        type=int, 
        default=2000,
        help="Number of prompts to process"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size (if not set, will use optimal settings based on GPU memory)"
    )
    
    return parser.parse_args()

def main():
    """主函数 - 使用SimPO模型运行RPS扰动实验（vLLM版本）"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置输出目录
    model_short_name = args.model_name.split("/")[-1]
    output_dir = os.path.join(args.output_dir, model_short_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Using model: {args.model_name}")
    
    # 为每个方向运行生成实验
    directions_to_test = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    
    print(f"\n🚀 开始生成SimPO RPS响应 (完整版本 with vLLM)")
    print(f"📐 将测试方向: {directions_to_test}")
    print(f"📊 每个方向测试{args.num_prompts}个prompts")
    print(f"🤖 使用模型: {args.model_name} (加速vLLM)")
    
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
        
        # 获取batch_size
        if args.batch_size is None:
            settings = get_optimal_settings()
            batch_size = settings["batch_size"]
        else:
            batch_size = args.batch_size
        
        print(f"📊 Using batch_size: {batch_size}")
        
        # 运行SimPO生成
        run_simpo_generation(
            result_dir=direction_dir,
            valid_vs=valid_vs,
            valid_angles=valid_angles,
            main_v=main_v,
            model_name=args.model_name,
            prompt_limit=args.num_prompts,
            batch_size=batch_size
        )
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(1)  # vLLM内存管理更高效，减少等待时间
    
    print("\n🎉 所有SimPO生成实验完成！")
    print(f"📁 生成结果保存在: {output_dir}")
    print(f"📂 目录结构:")
    for direction in directions_to_test:
        direction_path = os.path.join(output_dir, direction)
        if os.path.exists(direction_path):
            files = os.listdir(direction_path)
            print(f"  {direction}/: {len(files)} 个文件")
            for file in files[:3]:  # 显示前3个文件
                file_path = os.path.join(direction_path, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"    - {file} ({file_size:.1f} KB)")
    
    # 🔄 服务器使用提示
    print(f"\n💡 Results are saved to: {output_dir}")
    print(f"💡 You can find CSV files for each direction in the output directory")

# 运行主函数
if __name__ == "__main__":
    main()
