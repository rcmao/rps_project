#!/bin/bash

# best_nemotron_solution.sh - 最佳Nemotron-3-8B-SteerLM加载方案
echo "🚀 Best Solution for Nemotron-3-8B-SteerLM"
echo "📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm"

# 检查Docker服务
if ! systemctl is-active --quiet docker; then
    echo "🐳 Starting Docker service..."
    systemctl start docker
    systemctl enable docker
fi

# 创建模型目录
mkdir -p /root/nemotron_models
cd /root/nemotron_models

# 下载模型（如果还没有）
if [ ! -f "Nemotron-3-8B-Chat-4k-SteerLM.nemo" ]; then
    echo "📥 Downloading Nemotron model..."
    wget -O Nemotron-3-8B-Chat-4k-SteerLM.nemo \
        "https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm/resolve/main/Nemotron-3-8B-Chat-4k-SteerLM.nemo"
fi

# 创建官方格式的测试脚本
cat > test_nemotron_official.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import torch
import time

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        
        # 测试不同的prompt
        test_cases = [
            {
                "prompt": "Write a poem about NVIDIA in the style of Shakespeare",
                "description": "Creative Poem",
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
                "description": "Educational Explanation",
                "attributes": {
                    "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                    "complexity": 2, "verbosity": 3, "toxicity": 0, "humor": 0,
                    "creativity": 1, "violence": 0, "helpfulness": 4,
                    "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                    "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"🧪 Test {i+1}: {test_case['description']}")
            print(f"{'='*60}")
            
            prompt, attr_string = build_official_prompt(test_case["prompt"], test_case["attributes"])
            
            print(f"📝 Prompt: {test_case['prompt']}")
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
            
            print(f"\n📊 Attribute string used: {attr_string}")
        
        print("\n✅ NemoQuery API test completed!")
        return True
        
    except ImportError:
        print("❌ NemoQuery not available.")
        return False
    except Exception as e:
        print(f"❌ NemoQuery failed: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Starting Nemotron-3-8B-SteerLM test with official approach!")
    
    if test_with_nemo_query():
        print("\n✅ Success! Nemotron model is working correctly!")
        print("🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
        print("📚 Framework: NVIDIA NeMo")
        print("📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    else:
        print("\n❌ Failed to test with NemoQuery API")
        print("💡 Please ensure the NeMo inference server is running")

if __name__ == "__main__":
    main()
EOF

# 创建启动NeMo推理服务器的脚本
cat > start_nemo_server.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting NeMo inference server..."

# 停止现有容器
docker stop nemotron-server 2>/dev/null || true
docker rm nemotron-server 2>/dev/null || true

# 启动NeMo推理服务器
docker run -d --name nemotron-server \
    --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    -v /root/nemotron_models:/workspace \
    nvcr.io/nvidia/nemo:25.02 \
    python -m nemo.deploy.inference.server \
    --model-path /workspace/Nemotron-3-8B-Chat-4k-SteerLM.nemo \
    --port 8000

echo "⏳ Waiting for server to start..."
sleep 30

echo "✅ NeMo inference server started!"
echo "🌐 Server URL: http://localhost:8000"
echo "🧪 Run test: python3 test_nemotron_official.py"
EOF

chmod +x start_nemo_server.sh
chmod +x test_nemotron_official.py

echo "✅ Best solution created!"
echo ""
echo "📋 Usage Instructions:"
echo "1. Start NeMo server: bash start_nemo_server.sh"
echo "2. Run test: python3 test_nemotron_official.py"
echo "3. Clean up: docker stop nemotron-server"
echo ""
echo "🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm"
echo "📚 Framework: NVIDIA NeMo"
echo "📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm" 