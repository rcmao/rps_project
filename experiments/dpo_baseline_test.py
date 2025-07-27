# dpo_baseline_test.py
# 快速测试版本 - 验证脚本功能

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
import time
import random

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/root/autodl-tmp/hf_cache'

print("🌏 已设置Hugging Face国内镜像")
print("💾 模型缓存目录: /root/autodl-tmp/hf_cache")

# 简化的方向定义 - 只测试一个方向
TEST_DIRECTIONS = {
    "v3": {"vector": (0.9848, 0.1736), "angle": 10},
}

def setup_environment():
    """设置环境"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    return device

def load_models_simple(device):
    """简化的模型加载"""
    cache_dir = '/root/autodl-tmp/hf_cache'
    
    print("🤖 Loading DPO model...")
    # 使用更简单的加载方式
    dpo_model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/stablelm-zephyr-3b",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    print("✅ DPO model loaded!")
    
    dpo_tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/stablelm-zephyr-3b",
        cache_dir=cache_dir
    )
    if dpo_tokenizer.pad_token_id is None:
        dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
    print("✅ DPO tokenizer loaded!")
    
    print("🏆 Loading Reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
        torch_dtype=torch.float32,
        device_map="cpu",
        cache_dir=cache_dir,
    )
    print("✅ Reward model loaded on CPU!")
    
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
        cache_dir=cache_dir
    )
    print("✅ Reward tokenizer loaded!")
    
    return dpo_model, dpo_tokenizer, reward_model, reward_tokenizer

def build_dpa_input(prompt, v1, v2):
    """构造DPA输入"""
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

def generate_single_response(prompt, prompt_id, direction_info, dpo_model, dpo_tokenizer, device):
    """生成单个响应 - 简化版本"""
    try:
        v1, v2 = direction_info["vector"]
        messages = build_dpa_input(prompt, v1, v2)
        
        # 简单的tokenization
        text = dpo_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = dpo_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = inputs.to(device)
        
        print(f"    🔄 Generating response for prompt {prompt_id}...")
        
        with torch.no_grad():
            outputs = dpo_model.generate(
                **inputs,
                max_new_tokens=256,  # 更短的响应
                temperature=0.7,
                do_sample=True,
                pad_token_id=dpo_tokenizer.pad_token_id,
            )
        
        # 解码
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = dpo_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response": response,
            "direction_vector": f"({v1:.4f}, {v2:.4f})",
        }
    
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return None

def score_response_simple(prompt, response, reward_model, reward_tokenizer, v1, v2):
    """简化的评分"""
    try:
        template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        inputs = reward_tokenizer(
            template.format(prompt=prompt, response=response),
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        with torch.no_grad():
            logits = reward_model(**inputs).logits.squeeze().cpu().numpy()
        
        helpfulness = logits[9]
        verbosity = logits[4]
        dpa_score = v1 * helpfulness + v2 * verbosity
        
        return {
            "helpfulness": float(helpfulness),
            "verbosity": float(verbosity),
            "dpa_score": float(dpa_score)
        }
    
    except Exception as e:
        print(f"❌ Error scoring: {e}")
        return {"helpfulness": 0.0, "verbosity": 0.0, "dpa_score": 0.0}

def main():
    """主函数 - 测试版"""
    print("🧪 DPO Baseline Test - 快速验证版本")
    
    # 设置环境
    device = setup_environment()
    
    # 加载数据集 - 只用5个prompts测试
    print("📦 Loading test prompts...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        prompts = ds["prompt"][:5]  # 只测试5个
        prompt_ids = list(range(len(prompts)))
        print(f"✅ Loaded {len(prompts)} test prompts")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 加载模型
    dpo_model, dpo_tokenizer, reward_model, reward_tokenizer = load_models_simple(device)
    
    # 测试单个方向
    direction_name = "v3"
    direction_info = TEST_DIRECTIONS[direction_name]
    v1, v2 = direction_info["vector"]
    
    print(f"\n🎯 Testing direction {direction_name}: {direction_info['vector']}")
    
    results = []
    
    # 逐个处理prompts
    for i, (prompt, prompt_id) in enumerate(zip(prompts, prompt_ids)):
        print(f"\n📝 Processing prompt {i+1}/{len(prompts)} (ID: {prompt_id})")
        print(f"Prompt preview: {prompt[:100]}...")
        
        # 生成响应
        response_data = generate_single_response(
            prompt, prompt_id, direction_info, dpo_model, dpo_tokenizer, device
        )
        
        if response_data:
            print(f"✅ Generated response: {response_data['response'][:100]}...")
            
            # 评分
            scores = score_response_simple(
                prompt, response_data['response'], reward_model, reward_tokenizer, v1, v2
            )
            
            response_data.update(scores)
            results.append(response_data)
            
            print(f"📊 DPA Score: {scores['dpa_score']:.2f}")
        else:
            print("❌ Failed to generate response")
    
    # 保存结果
    if results:
        output_file = "/root/rps/data/dpo_test_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Test completed! Results saved to {output_file}")
        print(f"📊 Generated {len(results)} responses successfully")
    else:
        print("\n❌ No responses generated")

if __name__ == "__main__":
    main() 