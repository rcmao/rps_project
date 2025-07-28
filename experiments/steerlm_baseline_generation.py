# steerlm_nemotron_baseline.py - 使用Nemotron模型进行实验
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
import time
import random
import json

# =============================================================================
# 🌐 网络配置 - 支持国内镜像
# =============================================================================

# 国内镜像选项
MIRROR_OPTIONS = {
    "official": {
        "hf_endpoint": None,
        "name": "Official HuggingFace",
        "description": "直接访问官方HuggingFace"
    },
    "hf_mirror": {
        "hf_endpoint": "https://hf-mirror.com", 
        "name": "HF Mirror",
        "description": "国内HF镜像 (推荐)"
    },
    "modelfun": {
        "hf_endpoint": "https://www.modelfun.cn",
        "name": "ModelFun",
        "description": "模型乐园镜像"
    }
}

def setup_mirror(mirror_choice="hf_mirror"):
    """设置镜像配置"""
    if mirror_choice in MIRROR_OPTIONS:
        mirror = MIRROR_OPTIONS[mirror_choice]
        if mirror["hf_endpoint"]:
            os.environ['HF_ENDPOINT'] = mirror["hf_endpoint"]
        elif 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        
        print(f"🌐 使用镜像: {mirror['name']} - {mirror['description']}")
        return mirror["name"]
    else:
        print(f"❌ 未知镜像选择: {mirror_choice}")
        return "Unknown"

# 设置网络超时和重试
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'

# 默认使用国内镜像
current_mirror = setup_mirror("hf_mirror")  # 可改为 "official" 或 "modelfun"

print("🌐 Network configuration:")
print(f"Current Mirror: {current_mirror}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Official HuggingFace')}")
print(f"Cache dir: {os.environ.get('TRANSFORMERS_CACHE', 'Default')}")

# =============================================================================

# 定义方向向量
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

def load_nemotron_steerlm(device):
    """加载Nemotron模型 - 使用可用的HuggingFace格式版本"""
    print("🤖 Loading Nemotron model...")
    
    # 使用可用的Nemotron模型
    model_options = [
        "nvidia/nemotron-3-8b-base-4k-hf",  # Base model - HF格式
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",  # 新的Nemotron模型
        "nvidia/Nemotron-4-340B-Instruct"  # 更大的模型
    ]
    
    for model_name in model_options:
        try:
            print(f"🔄 尝试加载模型: {model_name}")
            
            # 尝试加载模型
            steerlm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/root/.cache/huggingface",
                force_download=False,
                resume_download=True
            )
            steerlm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/root/.cache/huggingface",
                force_download=False,
                resume_download=True
            )
            
            # 设置pad_token
            if steerlm_tokenizer.pad_token is None:
                steerlm_tokenizer.pad_token = steerlm_tokenizer.eos_token
            
            print(f"✅ 成功加载模型: {model_name}")
            break
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 加载失败: {str(e)[:100]}...")
            continue
    
    else:
        raise Exception("所有模型都加载失败")
    
    # 加载reward模型
    print("🏆 Loading reward model...")
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1", 
            trust_remote_code=True,
            cache_dir="/root/.cache/huggingface"
        ).to(device)
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            cache_dir="/root/.cache/huggingface"
        )
        print("✅ Reward model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load reward model: {e}")
        reward_model, reward_tokenizer = None, None
    
    return steerlm_model, steerlm_tokenizer, reward_model, reward_tokenizer

def dpa_vector_to_nemotron_attributes(v1, v2):
    """映射DPA向量到Nemotron属性格式"""
    helpfulness = max(0, min(4, round(v1 * 4)))
    verbosity = max(0, min(4, round(v2 * 4)))
    
    return {
        "quality": 4,
        "understanding": 4,
        "correctness": 4,
        "coherence": 4,
        "complexity": 4,
        "verbosity": verbosity,
        "toxicity": 0,
        "humor": 0,
        "creativity": 0,
        "violence": 0,
        "helpfulness": helpfulness,
        "not_appropriate": 0,
        "hate_speech": 0,
        "sexual_content": 0,
        "fails_task": 0,
        "political_content": 0,
        "moral_judgement": 0,
        "lang": "en"
    }

def build_nemotron_prompt(prompt, v1, v2, model_name=""):
    """构建适配不同Nemotron模型的prompt格式"""
    
    # 判断模型类型并构建相应格式的prompt
    if "steerlm" in model_name.lower():
        # SteerLM格式 (如果有的话)
        attrs = dpa_vector_to_nemotron_attributes(v1, v2)
    attr_string = f"quality:{attrs['quality']},understanding:{attrs['understanding']},correctness:{attrs['correctness']},coherence:{attrs['coherence']},complexity:{attrs['complexity']},verbosity:{attrs['verbosity']},toxicity:{attrs['toxicity']},humor:{attrs['humor']},creativity:{attrs['creativity']},violence:{attrs['violence']},helpfulness:{attrs['helpfulness']},not_appropriate:{attrs['not_appropriate']},hate_speech:{attrs['hate_speech']},sexual_content:{attrs['sexual_content']},fails_task:{attrs['fails_task']},political_content:{attrs['political_content']},moral_judgement:{attrs['moral_judgement']},lang:{attrs['lang']}"
    
    nemotron_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    return nemotron_prompt, attr_string

    elif "llama" in model_name.lower():
        # Llama-Nemotron格式
        # 使用reasoning on/off来控制详细程度
        reasoning_mode = "on" if v1 > 0.5 else "off"
        
        nemotron_prompt = f"""<|start_header_id|>system<|end_header_id|>

detailed thinking {reasoning_mode}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return nemotron_prompt, f"reasoning={reasoning_mode}"
        
    else:
        # 标准Nemotron格式
        nemotron_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
"""
        return nemotron_prompt, "standard_format"

def generate_responses_for_direction(prompts, v1, v2, steerlm_model, steerlm_tokenizer, device, direction_name, model_name=""):
    """为指定方向生成响应"""
    print(f"🎯 Generating responses for {direction_name} (v1={v1:.4f}, v2={v2:.4f})")
    
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating {direction_name}")):
        prompt_results = []
        
        # 为每个prompt生成3个响应
        for sample_id in range(3):
            try:
                nemotron_prompt, attr_string = build_nemotron_prompt(prompt, v1, v2, model_name)
                
                # 生成响应
                inputs = steerlm_tokenizer(nemotron_prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
                
                with torch.no_grad():
                    outputs = steerlm_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=steerlm_tokenizer.pad_token_id
                    )
                
                response = steerlm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # 清理输出（移除特殊标记）
                if "llama" in model_name.lower():
                    # Llama格式清理
                    response = response.split("<|eot_id|>")[0].strip()
                else:
                    # Nemotron格式清理
                response = response.split("<extra_id_1>")[0].strip()
                
                prompt_results.append({
                    "prompt_id": i,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "response": response,
                    "direction": direction_name,
                    "v1": v1,
                    "v2": v2,
                    "attributes": attr_string,
                    "model_name": model_name
                })
                
            except Exception as e:
                print(f"❌ Error generating response for prompt {i}, sample {sample_id}: {e}")
                prompt_results.append({
                    "prompt_id": i,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "response": f"ERROR: {str(e)}",
                    "direction": direction_name,
                    "v1": v1,
                    "v2": v2,
                    "attributes": "ERROR",
                    "model_name": model_name
                })
        
        results.extend(prompt_results)
    
    return results

def main():
    """主函数 - 测试Nemotron模型"""
    print("🚀 Starting Nemotron experiment!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据显存设置batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory >= 80:  # A100 80GB
            batch_size = 4
        elif gpu_memory >= 48:  # L40S 48GB
            batch_size = 2
        elif gpu_memory >= 24:  # RTX 4090等
            batch_size = 1
        else:
            batch_size = 1
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB, using batch_size={batch_size}")
    else:
        batch_size = 1
        print("💻 Using CPU, batch_size=1")
    
    # 加载数据
    print("📦 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:10]  # 先测试10个
    
    # 加载模型
    try:
        steerlm_model, steerlm_tokenizer, reward_model, reward_tokenizer = load_nemotron_steerlm(device)
        # 获取实际加载的模型名称
        model_name = steerlm_model.config.name_or_path if hasattr(steerlm_model.config, 'name_or_path') else "unknown"
        print(f"📝 使用模型: {model_name}")
    except Exception as e:
        print("💡 模型加载失败，可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保已登录HuggingFace: huggingface-cli login")
        print("3. 访问模型页面接受许可证")
        return
    
    # 测试一个prompt
    test_prompt = prompts[0]
    v1, v2 = 0.8, 0.4  # 高helpfulness，中等verbosity
    
    print(f"\n🧪 Testing with prompt: {test_prompt[:100]}...")
    print(f"🎯 DPA vector: ({v1}, {v2})")
    
    nemotron_prompt, attr_string = build_nemotron_prompt(test_prompt, v1, v2, model_name)
    
    print(f"\n📝 Generated prompt:")
    print(f"```\n{nemotron_prompt}\n```")
    
    # 生成响应
    inputs = steerlm_tokenizer(nemotron_prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    
    with torch.no_grad():
        outputs = steerlm_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=steerlm_tokenizer.pad_token_id
        )
    
    response = steerlm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 清理输出
    if "llama" in model_name.lower():
        response = response.split("<|eot_id|>")[0].strip()
    else:
    response = response.split("<extra_id_1>")[0].strip()
    
    print(f"\n🎯 Model Response:")
    print(f"```\n{response}\n```")
    
    print(f"\n✅ Nemotron model test successful!")
    print(f"📊 Control attributes: {attr_string}")
    print(f"🔧 Model used: {model_name}")

if __name__ == "__main__":
    main()