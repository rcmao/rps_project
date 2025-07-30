# final_nemotron_solution.py - 最终Nemotron-3-8B-SteerLM解决方案
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

def demonstrate_steerlm_format():
    """演示SteerLM格式"""
    print("🚀 Demonstrating Nemotron-3-8B-SteerLM format!")
    print("📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    
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
        },
        {
            "prompt": "Tell me a joke about programming",
            "description": "Humorous Response (High Humor)",
            "attributes": {
                "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                "complexity": 2, "verbosity": 3, "toxicity": 0, "humor": 4,
                "creativity": 3, "violence": 0, "helpfulness": 3,
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
        
        # 显示属性映射
        print(f"\n🎯 Attribute Mapping:")
        for key, value in attributes.items():
            if isinstance(value, (int, float)) and value > 0:  # 只显示非零属性
                print(f"  {key}: {value}/4")
        
        print(f"\n💡 This demonstrates the official SteerLM format for Nemotron-3-8B-SteerLM")
        print(f"🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
        print(f"📚 Framework: NVIDIA NeMo")
        print(f"📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    
    print(f"\n{'='*60}")
    print("📋 Summary")
    print("="*60)
    print("✅ Successfully demonstrated Nemotron-3-8B-SteerLM format!")
    print("🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
    print("📚 Framework: NVIDIA NeMo")
    print("📖 Documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    print("💡 Note: For actual model inference, use the official NeMo Docker container")
    print("🐳 Docker command: docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02")

def create_usage_guide():
    """创建使用指南"""
    guide = """
# Nemotron-3-8B-SteerLM 使用指南

## 🎯 最佳方案

根据官方文档，推荐使用以下方法：

### 方案1：官方 NeMo Docker 容器（推荐）
```bash
# 启动官方 NeMo 容器
docker run --gpus all -it --rm nvcr.io/nvidia/nemo:25.02

# 在容器内运行
python -c "
from nemo.deploy import NemoQuery
nq = NemoQuery(url='localhost:8000', model_name='Nemotron-3-8B-Chat-4K-SteerLM')
# 使用官方 SteerLM 格式
"
```

### 方案2：NeMo 推理服务器
```bash
# 启动推理服务器
docker run -d --gpus all -p 8000:8000 nvcr.io/nvidia/nemo:25.02 \\
    python -m nemo.deploy.inference.server \\
    --model-path /path/to/Nemotron-3-8B-Chat-4k-SteerLM.nemo \\
    --port 8000
```

## 📝 官方 SteerLM 格式

```python
PROMPT_TEMPLATE = \"\"\"<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>quality:4,understanding:4,correctness:4,coherence:4,complexity:4,verbosity:4,toxicity:0,humor:0,creativity:0,violence:0,helpfulness:4,not_appropriate:0,hate_speech:0,sexual_content:0,fails_task:0,political_content:0,moral_judgement:0,lang:en\"\"\"
```

## 🎛️ 属性控制

每个属性可以设置为 0-4 的值：
- quality: 响应质量
- understanding: 理解程度
- correctness: 正确性
- coherence: 连贯性
- complexity: 复杂度
- verbosity: 详细程度
- toxicity: 毒性
- humor: 幽默感
- creativity: 创造性
- violence: 暴力内容
- helpfulness: 有用性
- not_appropriate: 不当内容
- hate_speech: 仇恨言论
- sexual_content: 性内容
- fails_task: 任务失败
- political_content: 政治内容
- moral_judgement: 道德判断

## 📚 参考资料

- 官方文档: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm
- NeMo 框架: https://github.com/NVIDIA/NeMo
- SteerLM 论文: https://arxiv.org/abs/2310.05344
"""
    
    with open('nemotron_usage_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Usage guide created: nemotron_usage_guide.md")

def main():
    """主函数"""
    print("🚀 Final Nemotron-3-8B-SteerLM Solution")
    print("📖 Based on official documentation: https://huggingface.co/nvidia/nemotron-3-8b-chat-4k-steerlm")
    
    # 演示 SteerLM 格式
    demonstrate_steerlm_format()
    
    # 创建使用指南
    create_usage_guide()
    
    print("\n✅ Solution completed!")
    print("📖 Check nemotron_usage_guide.md for detailed instructions")

if __name__ == "__main__":
    main() 