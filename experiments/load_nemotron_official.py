# load_nemotron_official.py - 使用官方NeMo框架加载Nemotron-3-8B-SteerLM
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

def download_nemotron_model():
    """下载Nemotron-3-8B-SteerLM模型"""
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

def build_official_prompt(prompt, attributes=None):
    """构建官方SteerLM格式的prompt"""
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

def test_with_nemo_query():
    """使用官方NemoQuery API测试"""
    print("🚀 Testing with official NemoQuery API...")
    
    try:
        from nemo.deploy import NemoQuery
        
        # 连接到NeMo推理服务器
        nq = NemoQuery(url="localhost:8000", model_name="Nemotron-3-8B-Chat-4K-SteerLM")
        
        # 测试prompt
        test_prompt = "Write a poem about NVIDIA in the style of Shakespeare"
        prompt, attr_string = build_official_prompt(test_prompt)
        
        print(f"📝 Test prompt: {test_prompt}")
        print(f"📝 Generated prompt:")
        print(f"```\n{prompt}\n```")
        
        # 使用官方API生成响应
        output = nq.query_llm(
            prompts=[prompt], 
            max_output_token=200, 
            top_k=1, 
            top_p=0.0, 
            temperature=0.1
        )
        
        # 后处理输出（官方文档要求）
        output = [[s.split("<extra_id_1>", 1)[0].strip() for s in out] for out in output]
        
        print(f"\n🎯 Generated Response:")
        print(f"```\n{output[0][0]}\n```")
        
        print("✅ NemoQuery API test completed!")
        return True
        
    except ImportError:
        print("❌ NemoQuery not available. Trying alternative approach...")
        return False
    except Exception as e:
        print(f"❌ NemoQuery failed: {e}")
        return False

def test_with_direct_nemo():
    """直接使用NeMo框架测试"""
    print("🚀 Testing with direct NeMo framework...")
    
    try:
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from lightning.pytorch import Trainer
        
        # 下载模型
        model_path = download_nemotron_model()
        if model_path is None:
            return False
        
        print(f"🤖 Loading model from: {model_path}")
        
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
        print("✅ Model loaded successfully!")
        
        # 测试prompt
        test_prompt = "Write a poem about NVIDIA in the style of Shakespeare"
        prompt, attr_string = build_official_prompt(test_prompt)
        
        print(f"\n📝 Test prompt: {test_prompt}")
        print(f"📝 Generated prompt:")
        print(f"```\n{prompt}\n```")
        
        # 生成响应
        print("\n⚡ Generating response...")
        start_time = time.time()
        
        length_params = {"max_length": 512, "min_length": 1}
        sampling_params = {
            "use_greedy": False,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        
        response = model.generate([prompt], length_params, sampling_params)
        
        end_time = time.time()
        
        if response and len(response) > 0:
            # 清理输出
            if response[0].startswith(prompt):
                response = response[0][len(prompt):].strip()
            else:
                response = response[0]
            
            response = response.split("<extra_id_1>")[0].strip()
            
            print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
            print(f"```\n{response}\n```")
            
            print(f"\n📊 Attribute string used: {attr_string}")
            print("✅ Direct NeMo test completed!")
            return True
        else:
            print("❌ Empty response from model")
            return False
            
    except Exception as e:
        print(f"❌ Direct NeMo failed: {e}")
        return False

def create_docker_solution():
    """创建Docker解决方案"""
    print("🐳 Creating Docker solution...")
    
    docker_script = '''#!/bin/bash

# 启动NeMo推理服务器
echo "🚀 Starting NeMo inference server..."
docker run -d --name nemotron-server \\
    --gpus all \\
    --shm-size=16g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -p 8000:8000 \\
    nvcr.io/nvidia/nemo:25.02 \\
    python -m nemo.deploy.inference.server \\
    --model-path /workspace/Nemotron-3-8B-Chat-4k-SteerLM.nemo \\
    --port 8000

# 等待服务器启动
echo "⏳ Waiting for server to start..."
sleep 30

# 运行测试
echo "🧪 Running test..."
python3 load_nemotron_official.py

# 清理
echo "🧹 Cleaning up..."
docker stop nemotron-server
docker rm nemotron-server

echo "✅ Docker solution completed!"
'''
    
    with open('run_nemotron_docker_official.sh', 'w') as f:
        f.write(docker_script)
    
    os.chmod('run_nemotron_docker_official.sh', 0o755)
    print("✅ Docker script created: run_nemotron_docker_official.sh")

def main():
    """主函数 - 尝试多种方法加载Nemotron模型"""
    print("🚀 Starting Nemotron-3-8B-SteerLM loading test!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 方法1：尝试官方NemoQuery API
    print("\n" + "="*60)
    print("🧪 Method 1: Official NemoQuery API")
    print("="*60)
    
    if test_with_nemo_query():
        print("✅ Success with NemoQuery API!")
        return
    
    # 方法2：尝试直接NeMo框架
    print("\n" + "="*60)
    print("🧪 Method 2: Direct NeMo Framework")
    print("="*60)
    
    if test_with_direct_nemo():
        print("✅ Success with direct NeMo!")
        return
    
    # 方法3：创建Docker解决方案
    print("\n" + "="*60)
    print("🧪 Method 3: Docker Solution")
    print("="*60)
    
    create_docker_solution()
    print("💡 Docker solution created. Run: bash run_nemotron_docker_official.sh")
    
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    print("🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
    print("📚 Framework: NVIDIA NeMo")
    print("📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    print("💡 Best approach: Use Docker container with official NeMo framework")

if __name__ == "__main__":
    main() 