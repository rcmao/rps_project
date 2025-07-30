# steerlm_demo.py - SteerLM格式演示（使用Transformers模型）
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset

# =============================================================================
# 🌐 网络配置 - 支持国内镜像
# =============================================================================

# 设置网络超时和重试
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("🌐 Network configuration:")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'Official HuggingFace')}")
print(f"Cache dir: {os.environ.get('TRANSFORMERS_CACHE', 'Default')}")

# =============================================================================

def load_demo_model():
    """加载演示用的Transformers模型"""
    print("🤖 Loading demo model...")
    
    try:
        # 使用一个较小的模型进行演示
        model_name = "microsoft/DialoGPT-medium"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="/root/.cache/huggingface"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            cache_dir="/root/.cache/huggingface"
        )
        
        model.eval()
        print("✅ Demo model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load demo model: {e}")
        return None, None

def dpa_vector_to_steerlm_attributes(v1, v2):
    """映射DPA向量到SteerLM属性格式"""
    helpfulness = max(0, min(4, round(v1 * 4)))
    verbosity = max(0, min(4, round(v2 * 4)))
    
    return {
        "quality": 4,
        "understanding": 4,
        "correctness": 4,
        "coherence": 4,
        "complexity": 2,
        "verbosity": verbosity,
        "toxicity": 0,
        "humor": 0,
        "creativity": 1,
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

def build_simple_prompt(prompt, v1, v2):
    """构建简化的prompt格式用于演示"""
    helpfulness = max(0, min(4, round(v1 * 4)))
    verbosity = max(0, min(4, round(v2 * 4)))
    
    # 简化的prompt格式
    simple_prompt = f"""System: You are a helpful AI assistant. Please respond with helpfulness level {helpfulness}/4 and verbosity level {verbosity}/4.

User: {prompt}

Assistant:"""
    
    return simple_prompt

def generate_with_model(model, tokenizer, prompt_text, max_tokens=512, temperature=0.7):
    """使用模型生成文本"""
    try:
        # 编码输入
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        
        # 移动到正确的设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成配置
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
        
        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        # 移除输入部分，只保留生成的内容
        if generated_text.startswith(prompt_text):
            response = generated_text[len(prompt_text):].strip()
        else:
            response = generated_text
        
        return response
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def demo_steerlm_format():
    """演示SteerLM格式"""
    print("🚀 Starting SteerLM format demo!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 加载演示模型
    model, tokenizer = load_demo_model()
    if model is None or tokenizer is None:
        return
    
    # 加载测试数据
    print("📦 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:3]  # 测试3个prompt
    
    # 测试不同的DPA向量
    test_vectors = [
        (0.8, 0.2, "High Helpfulness, Low Verbosity"),
        (0.5, 0.5, "Medium Helpfulness, Medium Verbosity"),
        (0.2, 0.8, "Low Helpfulness, High Verbosity"),
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i+1}: {prompt[:100]}...")
        print(f"{'='*60}")
        
        for v1, v2, description in test_vectors:
            print(f"\n🎯 {description} (v1={v1:.1f}, v2={v2:.1f})")
            
            # 生成SteerLM格式的prompt
            steerlm_prompt, attr_string = build_steerlm_prompt(prompt, v1, v2)
            simple_prompt = build_simple_prompt(prompt, v1, v2)
            
            print(f"\n📝 SteerLM Attribute String:")
            print(f"```\n{attr_string}\n```")
            
            print(f"\n📝 Simple Prompt Format:")
            print(f"```\n{simple_prompt}\n```")
            
            # 生成响应
            print("⚡ Generating response...")
            response = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt_text=simple_prompt,
                max_tokens=256,
                temperature=0.7
            )
            
            print(f"\n🎯 Generated Response:")
            print(f"```\n{response}\n```")
    
    print(f"\n✅ SteerLM format demo completed!")
    print(f"🔧 Model: microsoft/DialoGPT-medium (demo model)")
    print(f"💡 Note: This demonstrates SteerLM prompt format. For real SteerLM models, use NeMo framework.")

if __name__ == "__main__":
    demo_steerlm_format() 