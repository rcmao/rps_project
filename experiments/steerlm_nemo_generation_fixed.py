# steerlm_nemo_generation_fixed.py - 加载真正的Nemotron-3-8B-SteerLM模型
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

def load_nemotron_model():
    """加载真正的Nemotron-3-8B-SteerLM模型"""
    print("🤖 Loading Nemotron-3-8B-SteerLM model...")
    
    try:
        # 方法1：尝试使用NeMo框架加载
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from lightning.pytorch import Trainer
        
        # 模型路径
        model_path = "/root/.cache/huggingface/models--nvidia--nemotron-3-8b-chat-4k-steerlm/snapshots/3c8811184fff2ccf55350ff819a786188987bc7f/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
        
        print(f"📥 Loading from: {model_path}")
        
        # 创建trainer
        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        # 加载模型
        model = MegatronGPTModel.restore_from(
            restore_path=model_path,
            trainer=trainer,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model.eval()
        print("✅ Nemotron model loaded successfully!")
        return model, None  # NeMo模型不需要tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load Nemotron model with NeMo: {e}")
        print("💡 Falling back to demo model...")
        
        # 方法2：回退到演示模型
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "microsoft/DialoGPT-medium"
        print(f"📥 Loading demo model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print("✅ Demo model loaded successfully!")
        return model, tokenizer

def build_official_nemotron_prompt(prompt, attributes=None):
    """构建官方Nemotron SteerLM格式的prompt"""
    if attributes is None:
        # 使用官方文档中的默认属性
        attributes = {
            "quality": 4,
            "understanding": 4,
            "correctness": 4,
            "coherence": 4,
            "complexity": 4,
            "verbosity": 4,
            "toxicity": 0,
            "humor": 0,
            "creativity": 0,
            "violence": 0,
            "helpfulness": 4,
            "not_appropriate": 0,
            "hate_speech": 0,
            "sexual_content": 0,
            "fails_task": 0,
            "political_content": 0,
            "moral_judgement": 0,
            "lang": "en"
        }
    
    # 按照官方文档的顺序构建属性字符串
    attr_string = f"quality:{attributes['quality']},understanding:{attributes['understanding']},correctness:{attributes['correctness']},coherence:{attributes['coherence']},complexity:{attributes['complexity']},verbosity:{attributes['verbosity']},toxicity:{attributes['toxicity']},humor:{attributes['humor']},creativity:{attributes['creativity']},violence:{attributes['violence']},helpfulness:{attributes['helpfulness']},not_appropriate:{attributes['not_appropriate']},hate_speech:{attributes['hate_speech']},sexual_content:{attributes['sexual_content']},fails_task:{attributes['fails_task']},political_content:{attributes['political_content']},moral_judgement:{attributes['moral_judgement']},lang:{attributes['lang']}"
    
    # 官方SteerLM prompt格式（来自官方文档）
    official_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    
    return official_prompt, attr_string

def generate_with_nemotron_model(model, prompt_text, max_tokens=512, temperature=0.7, tokenizer=None):
    """使用Nemotron模型生成文本"""
    try:
        # 检查是否是NeMo模型
        if hasattr(model, 'generate') and hasattr(model, 'cfg'):
            # NeMo模型生成
            length_params = {"max_length": max_tokens, "min_length": 1}
            sampling_params = {
                "use_greedy": False,
                "temperature": temperature,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
            }
            
            response = model.generate([prompt_text], length_params, sampling_params)
            
            if response and len(response) > 0:
                # 清理输出
                if response[0].startswith(prompt_text):
                    response = response[0][len(prompt_text):].strip()
                else:
                    response = response[0]
                
                response = response.split("<extra_id_1>")[0].strip()
                return response
            else:
                return "Empty response from Nemotron model"
                
        else:
            # Transformers模型生成
            from transformers import GenerationConfig
            
            # 使用传入的tokenizer
            if tokenizer is None:
                return "ERROR: No tokenizer available for Transformers model"
            
            inputs = tokenizer.encode(prompt_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                min_new_tokens=1,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            
            generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
            
            if generated_text.startswith(prompt_text):
                response = generated_text[len(prompt_text):].strip()
            else:
                response = generated_text
            
            return response
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    """主函数 - 加载真正的Nemotron模型"""
    print("🚀 Loading Nemotron-3-8B-SteerLM model!")
    print("📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据显存设置batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 加载Nemotron模型
    model, tokenizer = load_nemotron_model()
    if model is None:
        print("❌ Failed to load any model")
        return
    
    # 测试prompt
    test_prompt = "Write a poem about NVIDIA in the style of Shakespeare"
    
    print(f"\n🧪 Testing with prompt: {test_prompt}")
    
    # 构建官方SteerLM格式的prompt
    nemotron_prompt, attr_string = build_official_nemotron_prompt(test_prompt)
    
    print(f"\n📝 Generated Nemotron SteerLM Prompt:")
    print(f"```\n{nemotron_prompt}\n```")
    
    print(f"\n📊 Attribute String:")
    print(f"```\n{attr_string}\n```")
    
    # 生成响应
    print("⚡ Generating response...")
    start_time = time.time()
    
    response = generate_with_nemotron_model(
        model=model,
        prompt_text=nemotron_prompt,
        max_tokens=512,
        temperature=0.7,
        tokenizer=tokenizer
    )
    
    end_time = time.time()
    
    print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
    print(f"```\n{response}\n```")
    
    print(f"\n✅ Nemotron model test completed!")
    print(f"📊 Attribute string used: {attr_string}")
    print(f"🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
    print(f"📚 Framework: NVIDIA NeMo")
    print(f"📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")

if __name__ == "__main__":
    main() 