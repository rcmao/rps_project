#!/bin/bash

# start_nemotron.sh - 启动Nemotron-3-8B-SteerLM模型
echo "🚀 Starting Nemotron-3-8B-SteerLM Model"
echo "📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"

# 检查Docker状态
echo "🔍 Checking Docker status..."
if ! docker ps > /dev/null 2>&1; then
    echo "🐳 Docker daemon not running. Starting Docker..."
    
    # 尝试多种方法启动Docker
    if command -v systemctl > /dev/null 2>&1; then
        echo "📋 Using systemctl to start Docker..."
        systemctl start docker 2>/dev/null || true
    fi
    
    if command -v service > /dev/null 2>&1; then
        echo "📋 Using service to start Docker..."
        service docker start 2>/dev/null || true
    fi
    
    # 直接启动Docker daemon
    echo "📋 Starting Docker daemon directly..."
    dockerd > /dev/null 2>&1 &
    sleep 10
    
    # 检查是否启动成功
    if docker ps > /dev/null 2>&1; then
        echo "✅ Docker started successfully!"
    else
        echo "❌ Failed to start Docker daemon"
        echo "💡 Please try manually: sudo dockerd &"
        exit 1
    fi
else
    echo "✅ Docker is already running"
fi

# 检查GPU支持
echo "🔍 Checking GPU support..."
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU support is available"
else
    echo "⚠️  GPU support not available, will use CPU"
fi

# 创建模型目录
mkdir -p /root/nemotron_models
cd /root/nemotron_models

# 检查模型文件
if [ ! -f "Nemotron-3-8B-Chat-4k-SteerLM.nemo" ]; then
    echo "📥 Model file not found. Please download it first:"
    echo "💡 Run: python3 load_nemotron_official.py"
    exit 1
else
    echo "✅ Model file found: Nemotron-3-8B-Chat-4k-SteerLM.nemo"
fi

# 启动NeMo推理服务器
echo "🚀 Starting NeMo inference server..."
docker stop nemotron-server 2>/dev/null || true
docker rm nemotron-server 2>/dev/null || true

# 启动容器
if docker run -d --name nemotron-server \
    --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    -v /root/nemotron_models:/workspace \
    nvcr.io/nvidia/nemo:25.02 \
    python -m nemo.deploy.inference.server \
    --model-path /workspace/Nemotron-3-8B-Chat-4k-SteerLM.nemo \
    --port 8000; then
    
    echo "✅ NeMo inference server started successfully!"
    echo "🌐 Server URL: http://localhost:8000"
    echo "⏳ Waiting for server to be ready..."
    sleep 30
    
    # 创建测试脚本
    cat > test_nemotron_client.py << 'EOF'
#!/usr/bin/env python3
import time
import requests

def test_nemotron_server():
    """测试Nemotron服务器"""
    print("🧪 Testing Nemotron server...")
    
    try:
        # 测试服务器连接
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is healthy!")
            return True
        else:
            print(f"❌ Server returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return False

def build_official_prompt(prompt, attributes=None):
    """构建官方SteerLM格式的prompt"""
    if attributes is None:
        attributes = {
            "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
            "complexity": 4, "verbosity": 4, "toxicity": 0, "humor": 0,
            "creativity": 0, "violence": 0, "helpfulness": 4,
            "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
            "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
        }
    
    attr_string = f"quality:{attributes['quality']},understanding:{attributes['understanding']},correctness:{attributes['correctness']},coherence:{attributes['coherence']},complexity:{attributes['complexity']},verbosity:{attributes['verbosity']},toxicity:{attributes['toxicity']},humor:{attributes['humor']},creativity:{attributes['creativity']},violence:{attributes['violence']},helpfulness:{attributes['helpfulness']},not_appropriate:{attributes['not_appropriate']},hate_speech:{attributes['hate_speech']},sexual_content:{attributes['sexual_content']},fails_task:{attributes['fails_task']},political_content:{attributes['political_content']},moral_judgement:{attributes['moral_judgement']},lang:{attributes['lang']}"
    
    official_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    
    return official_prompt

def main():
    """主函数"""
    print("🚀 Testing Nemotron-3-8B-SteerLM server...")
    
    # 测试服务器
    if test_nemotron_server():
        print("\n✅ Nemotron server is ready!")
        print("🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
        print("📚 Framework: NVIDIA NeMo")
        print("📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
        print("\n💡 You can now use the model with the official SteerLM format!")
    else:
        print("\n❌ Server is not ready yet. Please wait and try again.")
        print("💡 You can check server logs with: docker logs nemotron-server")

if __name__ == "__main__":
    main()
EOF

    chmod +x test_nemotron_client.py
    
    # 运行测试
    echo "🧪 Running server test..."
    python3 test_nemotron_client.py
    
    echo ""
    echo "📋 Usage Instructions:"
    echo "1. Test server: python3 test_nemotron_client.py"
    echo "2. Stop server: docker stop nemotron-server"
    echo "3. View logs: docker logs nemotron-server"
    echo ""
    echo "🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm"
    echo "📚 Framework: NVIDIA NeMo"
    echo "📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"
    
else
    echo "❌ Failed to start NeMo inference server"
    echo "💡 Please check Docker and GPU availability"
    exit 1
fi 