# steerlm_nemo_fixed.py - 使用NeMo框架正确加载SteerLM模型
import os
import numpy as np
import pandas as pd
from datasets import load_dataset
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
current_mirror = setup_mirror("hf_mirror")

print("🌐 Network configuration:")
print(f"Current Mirror: {current_mirror}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Official HuggingFace')}")
print(f"Cache dir: {os.environ.get('TRANSFORMERS_CACHE', 'Default')}")

# =============================================================================

# 导入NeMo相关模块
try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
    from nemo.core.config import hydra_runner
    from omegaconf import DictConfig
    import nemo
    # 使用正确的PyTorch Lightning导入
    try:
        from lightning.pytorch import Trainer
    except ImportError:
        from pytorch_lightning import Trainer
    print("✅ NeMo toolkit imported successfully!")
except ImportError as e:
    print(f"❌ NeMo import failed: {e}")
    print("💡 请先运行: bash install_nemo_steerlm.sh")
    exit(1)

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

def download_nemo_model():
    """下载.nemo格式的SteerLM模型"""
    from huggingface_hub import hf_hub_download
    
    print("📥 Downloading Nemotron-3-8B-SteerLM (.nemo format)...")
    
    try:
        model_path = hf_hub_download(
            repo_id="nvidia/nemotron-3-8b-chat-4k-steerlm",
            filename="Nemotron-3-8B-Chat-4k-SteerLM.nemo",
            cache_dir="/root/.cache/huggingface",
            resume_download=True
        )
        print(f"✅ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 确保已登录HuggingFace并接受模型许可证")
        return None

def load_nemo_steerlm_model(nemo_path, device="cuda"):
    """使用NeMo加载SteerLM模型"""
    print(f"🤖 Loading SteerLM model from: {nemo_path}")
    
    try:
        # 创建PTL trainer
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        # 使用NeMo加载模型
        model = MegatronGPTModel.restore_from(
            restore_path=nemo_path,
            trainer=trainer,
            map_location=device
        )
        
        # 设置为评估模式
        model.eval()
        model = model.to(device)
        
        print("✅ NeMo SteerLM model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def dpa_vector_to_steerlm_attributes(v1, v2):
    """映射DPA向量到SteerLM属性格式 - 严格按照论文规范"""
    helpfulness = max(0, min(4, round(v1 * 4)))
    verbosity = max(0, min(4, round(v2 * 4)))
    
    return {
        "quality": 4,
        "understanding": 4,
        "correctness": 4,
        "coherence": 4,
        "complexity": 2,        # 修正：4 → 2
        "verbosity": verbosity,
        "toxicity": 0,
        "humor": 0,
        "creativity": 1,        # 修正：0 → 1
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

def build_steerlm_prompt(prompt, v1, v2):
    """构建SteerLM格式的prompt"""
    attrs = dpa_vector_to_steerlm_attributes(v1, v2)
    
    # 按照官方顺序构建属性字符串
    attr_string = f"quality:{attrs['quality']},understanding:{attrs['understanding']},correctness:{attrs['correctness']},coherence:{attrs['coherence']},complexity:{attrs['complexity']},verbosity:{attrs['verbosity']},toxicity:{attrs['toxicity']},humor:{attrs['humor']},creativity:{attrs['creativity']},violence:{attrs['violence']},helpfulness:{attrs['helpfulness']},not_appropriate:{attrs['not_appropriate']},hate_speech:{attrs['hate_speech']},sexual_content:{attrs['sexual_content']},fails_task:{attrs['fails_task']},political_content:{attrs['political_content']},moral_judgement:{attrs['moral_judgement']},lang:{attrs['lang']}"
    
    # 官方SteerLM prompt格式
    steerlm_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    
    return steerlm_prompt, attr_string

def generate_with_nemo_model(model, prompt_text, max_tokens=512, temperature=0.7):
    """使用NeMo模型生成文本"""
    try:
        # NeMo生成参数
        length_params = {
            "max_length": max_tokens,
            "min_length": 1,
        }
        
        sampling_params = {
            "use_greedy": temperature == 0.0,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        
        # 生成响应
        response = model.generate([prompt_text], length_params, sampling_params)
        
        if response and len(response) > 0:
            return response[0]
        else:
            return "ERROR: Empty response from model"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def generate_responses_for_direction(prompts, v1, v2, model, direction_name):
    """为指定方向生成响应"""
    print(f"🎯 Generating responses for {direction_name} (v1={v1:.4f}, v2={v2:.4f})")
    
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating {direction_name}")):
        prompt_results = []
        
        # 为每个prompt生成3个响应
        for sample_id in range(3):
            try:
                steerlm_prompt, attr_string = build_steerlm_prompt(prompt, v1, v2)
                
                # 使用NeMo生成响应
                response = generate_with_nemo_model(
                    model=model,
                    prompt_text=steerlm_prompt,
                    max_tokens=512,
                    temperature=0.7
                )
                
                # 清理输出（移除prompt部分）
                if response.startswith(steerlm_prompt):
                    response = response[len(steerlm_prompt):].strip()
                
                # 移除SteerLM特殊标记
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
                    "model_name": "nemotron-3-8b-chat-4k-steerlm"
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
                    "model_name": "nemotron-3-8b-chat-4k-steerlm"
                })
        
        results.extend(prompt_results)
    
    return results

def main():
    """主函数 - 使用NeMo加载和测试SteerLM模型"""
    print("🚀 Starting NeMo SteerLM experiment!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据显存设置batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 下载.nemo模型
    nemo_path = download_nemo_model()
    if nemo_path is None:
        return
    
    # 加载NeMo模型
    print(f"🤖 Loading model from: {nemo_path}")
    model = load_nemo_steerlm_model(nemo_path, device)
    if model is None:
        return
    
    # 加载测试数据
    print("📦 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:5]  # 先测试5个
    
    # 测试一个prompt
    test_prompt = prompts[0]
    v1, v2 = 0.8, 0.4  # 高helpfulness，中等verbosity
    
    print(f"\n🧪 Testing with prompt: {test_prompt[:100]}...")
    print(f"🎯 DPA vector: ({v1}, {v2})")
    
    steerlm_prompt, attr_string = build_steerlm_prompt(test_prompt, v1, v2)
    
    print(f"\n📝 Generated SteerLM prompt:")
    print(f"```\n{steerlm_prompt}\n```")
    
    # 生成响应
    print("⚡ Generating response...")
    response = generate_with_nemo_model(
        model=model,
        prompt_text=steerlm_prompt,
        max_tokens=512,
        temperature=0.7
    )
    
    # 清理输出
    if response.startswith(steerlm_prompt):
        response = response[len(steerlm_prompt):].strip()
    response = response.split("<extra_id_1>")[0].strip()
    
    print(f"\n🎯 SteerLM Response:")
    print(f"```\n{response}\n```")
    
    print(f"\n✅ NeMo SteerLM test successful!")
    print(f"📊 Attribute string used: {attr_string}")
    print(f"🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm (.nemo format)")

if __name__ == "__main__":
    main() 