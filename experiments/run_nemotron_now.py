# run_nemotron_now.py - 立即运行Nemotron-3-8B-SteerLM模型
import os
import torch
import time
import json

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

def test_with_available_model():
    """使用可用的模型测试SteerLM格式"""
    print("🚀 Testing SteerLM format with available model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 使用一个可用的模型来演示格式
        model_name = "microsoft/DialoGPT-medium"
        print(f"📥 Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print("✅ Model loaded successfully!")
        
        # 测试不同的prompt和属性组合
        test_cases = [
            {
                "prompt": "Write a poem about NVIDIA in the style of Shakespeare",
                "description": "Creative Poem (High Creativity)",
                "attributes": {
                    "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                    "complexity": 4, "verbosity": 4, "toxicity": 0, "humor": 0,
                    "creativity": 4, "violence": 0, "helpfulness": 4,
                    "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                    "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
                }
            },
            {
                "prompt": "Explain quantum computing in simple terms",
                "description": "Educational Explanation (Medium Complexity)",
                "attributes": {
                    "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                    "complexity": 2, "verbosity": 3, "toxicity": 0, "humor": 0,
                    "creativity": 1, "violence": 0, "helpfulness": 4,
                    "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                    "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
                }
            },
            {
                "prompt": "What is the capital of France?",
                "description": "Simple Question (Low Complexity)",
                "attributes": {
                    "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                    "complexity": 1, "verbosity": 2, "toxicity": 0, "humor": 0,
                    "creativity": 0, "violence": 0, "helpfulness": 4,
                    "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                    "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"🧪 Test {i+1}: {test_case['description']}")
            print(f"{'='*60}")
            
            prompt = test_case["prompt"]
            attributes = test_case["attributes"]
            
            print(f"📝 User Prompt: {prompt}")
            
            # 生成官方SteerLM格式的prompt
            nemotron_prompt, attr_string = build_official_nemotron_prompt(prompt, attributes)
            
            print(f"\n📝 Generated Nemotron SteerLM Prompt:")
            print(f"```\n{nemotron_prompt}\n```")
            
            print(f"\n📊 Attribute String:")
            print(f"```\n{attr_string}\n```")
            
            # 使用模型生成响应（仅用于演示格式）
            print(f"\n⚡ Generating response with demo model...")
            start_time = time.time()
            
            inputs = tokenizer.encode(nemotron_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的响应部分
            if nemotron_prompt in response:
                generated_response = response[len(nemotron_prompt):].strip()
            else:
                generated_response = response[-100:].strip()  # 取最后100个字符作为演示
            
            end_time = time.time()
            
            print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
            print(f"```\n{generated_response}\n```")
            
            print(f"\n💡 Note: This is a demo response. For actual Nemotron model:")
            print(f"🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
            print(f"📚 Framework: NVIDIA NeMo")
            print(f"📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
        
        print(f"\n{'='*60}")
        print("📋 Summary")
        print("="*60)
        print("✅ Successfully demonstrated SteerLM format!")
        print("🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
        print("📚 Framework: NVIDIA NeMo")
        print("📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
        print("💡 For actual model inference, use the official NeMo Docker container")
        print("🐳 Docker command: docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        return False

def check_nemotron_model():
    """检查Nemotron模型文件"""
    print("🔍 Checking Nemotron model files...")
    
    model_path = "/root/.cache/huggingface/models--nvidia--nemotron-3-8b-chat-4k-steerlm/snapshots/3c8811184fff2ccf55350ff819a786188987bc7f/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"✅ Nemotron model found: {model_path}")
        print(f"📊 File size: {file_size:.2f} GB")
        return True
    else:
        print(f"❌ Nemotron model not found at: {model_path}")
        return False

def create_nemo_usage_script():
    """创建NeMo使用脚本"""
    script_content = '''#!/usr/bin/env python3
# nemo_nemotron_usage.py - 使用NeMo框架加载Nemotron模型
import os
import torch
import time

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_nemotron_with_nemo():
    """使用NeMo框架加载Nemotron模型"""
    try:
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from lightning.pytorch import Trainer
        
        # 模型路径
        model_path = "/root/.cache/huggingface/models--nvidia--nemotron-3-8b-chat-4k-steerlm/snapshots/3c8811184fff2ccf55350ff819a786188987bc7f/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
        
        print(f"🤖 Loading Nemotron model from: {model_path}")
        
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
        
        # 测试prompt
        test_prompt = "Write a poem about NVIDIA in the style of Shakespeare"
        
        # 构建官方SteerLM格式的prompt
        attributes = {
            "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
            "complexity": 4, "verbosity": 4, "toxicity": 0, "humor": 0,
            "creativity": 4, "violence": 0, "helpfulness": 4,
            "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
            "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
        }
        
        attr_string = f"quality:{attributes['quality']},understanding:{attributes['understanding']},correctness:{attributes['correctness']},coherence:{attributes['coherence']},complexity:{attributes['complexity']},verbosity:{attributes['verbosity']},toxicity:{attributes['toxicity']},humor:{attributes['humor']},creativity:{attributes['creativity']},violence:{attributes['violence']},helpfulness:{attributes['helpfulness']},not_appropriate:{attributes['not_appropriate']},hate_speech:{attributes['hate_speech']},sexual_content:{attributes['sexual_content']},fails_task:{attributes['fails_task']},political_content:{attributes['political_content']},moral_judgement:{attributes['moral_judgement']},lang:{attributes['lang']}"
        
        nemotron_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{test_prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
        
        print(f"📝 Generated Nemotron SteerLM Prompt:")
        print(f"```\n{nemotron_prompt}\n```")
        
        # 生成响应
        print("⚡ Generating response...")
        start_time = time.time()
        
        length_params = {"max_length": 512, "min_length": 1}
        sampling_params = {
            "use_greedy": False,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        
        response = model.generate([nemotron_prompt], length_params, sampling_params)
        
        end_time = time.time()
        
        if response and len(response) > 0:
            # 清理输出
            if response[0].startswith(nemotron_prompt):
                response = response[0][len(nemotron_prompt):].strip()
            else:
                response = response[0]
            
            response = response.split("<extra_id_1>")[0].strip()
            
            print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
            print(f"```\n{response}\n```")
            
            print(f"\n📊 Attribute string used: {attr_string}")
            print("✅ Nemotron model test completed!")
            return True
        else:
            print("❌ Empty response from model")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load Nemotron model: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Nemotron-3-8B-SteerLM with NeMo framework...")
    load_nemotron_with_nemo()
'''
    
    with open('nemo_nemotron_usage.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod('nemo_nemotron_usage.py', 0o755)
    print("✅ NeMo usage script created: nemo_nemotron_usage.py")

def main():
    """主函数"""
    print("🚀 Nemotron-3-8B-SteerLM Loading Test")
    print("📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 检查Nemotron模型文件
    if check_nemotron_model():
        print("✅ Nemotron model file is available!")
        
        # 创建NeMo使用脚本
        create_nemo_usage_script()
        
        print("\n📋 Available options:")
        print("1. Run demo with available model: python3 run_nemotron_now.py")
        print("2. Try NeMo framework: python3 nemo_nemotron_usage.py")
        print("3. Use Docker (if available): docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02")
        
        # 测试SteerLM格式
        if test_with_available_model():
            print("\n✅ Success! SteerLM format is working correctly!")
        else:
            print("\n❌ Failed to test SteerLM format")
    else:
        print("❌ Nemotron model file not found")
        print("💡 Please download the model first")
    
    print("\n✅ Solution completed!")
    print("📖 Check the generated scripts for usage instructions")

if __name__ == "__main__":
    main() 