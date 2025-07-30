#!/bin/bash

# run_nemotron_docker.sh - 使用Docker运行Nemotron-3-8B-SteerLM模型
echo "🚀 Running Nemotron-3-8B-SteerLM with Docker..."

# 检查Docker是否可用
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    apt update
    apt install -y docker.io
    systemctl start docker
    systemctl enable docker
fi

# 检查NVIDIA Docker支持
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker not available. Installing NVIDIA Docker..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt update
    apt install -y nvidia-docker2
    systemctl restart docker
fi

# 创建模型目录
mkdir -p /root/nemotron_models
cd /root/nemotron_models

# 下载模型（如果还没有下载）
if [ ! -f "Nemotron-3-8B-Chat-4k-SteerLM.nemo" ]; then
    echo "📥 Downloading Nemotron model..."
    wget -O Nemotron-3-8B-Chat-4k-SteerLM.nemo \
        "https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm/resolve/main/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
fi

# 创建Python脚本用于测试
cat > test_nemotron.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import torch
import time

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from lightning.pytorch import Trainer
    print("✅ NeMo toolkit imported successfully!")
except ImportError as e:
    print(f"❌ NeMo import failed: {e}")
    sys.exit(1)

def load_model():
    """加载Nemotron模型"""
    model_path = "/workspace/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
    print(f"🤖 Loading model from: {model_path}")
    
    try:
        # 创建trainer
        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            precision=16,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        # 加载模型
        model = MegatronGPTModel.restore_from(
            restore_path=model_path,
            trainer=trainer,
            map_location="cuda"
        )
        
        model.eval()
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def build_prompt(prompt, attributes=None):
    """构建SteerLM格式的prompt"""
    if attributes is None:
        attributes = {
            "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
            "complexity": 4, "verbosity": 4, "toxicity": 0, "humor": 0,
            "creativity": 4, "violence": 0, "helpfulness": 4,
            "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
            "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
        }
    
    attr_string = ",".join([f"{k}:{v}" for k, v in attributes.items()])
    
    return f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""

def generate_response(model, prompt):
    """生成响应"""
    try:
        length_params = {"max_length": 512, "min_length": 1}
        sampling_params = {
            "use_greedy": False,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        
        response = model.generate([prompt], length_params, sampling_params)
        
        if response and len(response) > 0:
            return response[0]
        else:
            return "ERROR: Empty response"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    """主函数"""
    print("🚀 Starting Nemotron test...")
    
    # 加载模型
    model = load_model()
    if model is None:
        return
    
    # 测试prompt
    test_prompt = "Write a poem about NVIDIA in the style of Shakespeare"
    print(f"\n📝 Test prompt: {test_prompt}")
    
    # 构建SteerLM格式的prompt
    nemotron_prompt = build_prompt(test_prompt)
    print(f"\n📝 Generated prompt:")
    print(f"```\n{nemotron_prompt}\n```")
    
    # 生成响应
    print("\n⚡ Generating response...")
    start_time = time.time()
    
    response = generate_response(model, nemotron_prompt)
    
    end_time = time.time()
    
    # 清理输出
    if response.startswith(nemotron_prompt):
        response = response[len(nemotron_prompt):].strip()
    response = response.split("<extra_id_1>")[0].strip()
    
    print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
    print(f"```\n{response}\n```")
    
    print("\n✅ Nemotron test completed!")

if __name__ == "__main__":
    main()
EOF

# 运行Docker容器
echo "🐳 Running Nemotron in Docker container..."
docker run --rm -it \
    --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /root/nemotron_models:/workspace \
    nvcr.io/nvidia/nemo:25.02 \
    python /workspace/test_nemotron.py

echo "✅ Docker container completed!" 