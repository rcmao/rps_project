# load_nemotron_steerlm.py - 使用NeMo框架加载Nemotron-3-8B-SteerLM模型
import os
import torch
import time

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
    print("💡 请先运行: bash install_nemo_china_mirror.sh")
    exit(1)

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

def load_nemotron_model(nemo_path, device="cuda"):
    """使用NeMo加载Nemotron模型"""
    print(f"🤖 Loading Nemotron model from: {nemo_path}")
    
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
        
        print("✅ Nemotron model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def build_nemotron_prompt(prompt, attributes=None):
    """构建Nemotron SteerLM格式的prompt"""
    if attributes is None:
        # 默认属性设置
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
    
    # 按照官方顺序构建属性字符串
    attr_string = f"quality:{attributes['quality']},understanding:{attributes['understanding']},correctness:{attributes['correctness']},coherence:{attributes['coherence']},complexity:{attributes['complexity']},verbosity:{attributes['verbosity']},toxicity:{attributes['toxicity']},humor:{attributes['humor']},creativity:{attributes['creativity']},violence:{attributes['violence']},helpfulness:{attributes['helpfulness']},not_appropriate:{attributes['not_appropriate']},hate_speech:{attributes['hate_speech']},sexual_content:{attributes['sexual_content']},fails_task:{attributes['fails_task']},political_content:{attributes['political_content']},moral_judgement:{attributes['moral_judgement']},lang:{attributes['lang']}"
    
    # 官方Nemotron SteerLM prompt格式
    nemotron_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    
    return nemotron_prompt, attr_string

def generate_with_nemotron(model, prompt_text, max_tokens=512, temperature=0.7):
    """使用Nemotron模型生成文本"""
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

def test_nemotron_model():
    """测试Nemotron模型"""
    print("🚀 Starting Nemotron-3-8B-SteerLM test!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🚀 GPU Memory: {gpu_memory:.1f}GB")
    else:
        print("💻 Using CPU")
    
    # 下载模型
    nemo_path = download_nemotron_model()
    if nemo_path is None:
        return
    
    # 加载模型
    print(f"🤖 Loading model from: {nemo_path}")
    model = load_nemotron_model(nemo_path, device)
    if model is None:
        return
    
    # 测试不同的prompt和属性
    test_cases = [
        {
            "prompt": "Write a poem about NVIDIA in the style of Shakespeare",
            "attributes": {
                "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                "complexity": 4, "verbosity": 4, "toxicity": 0, "humor": 0,
                "creativity": 4, "violence": 0, "helpfulness": 4,
                "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
            },
            "description": "Creative Poem (High Creativity)"
        },
        {
            "prompt": "Explain quantum computing in simple terms",
            "attributes": {
                "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                "complexity": 2, "verbosity": 3, "toxicity": 0, "humor": 0,
                "creativity": 1, "violence": 0, "helpfulness": 4,
                "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
            },
            "description": "Educational Explanation (Medium Complexity)"
        },
        {
            "prompt": "What is the capital of France?",
            "attributes": {
                "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4,
                "complexity": 1, "verbosity": 2, "toxicity": 0, "humor": 0,
                "creativity": 0, "violence": 0, "helpfulness": 4,
                "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
                "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
            },
            "description": "Simple Question (Low Complexity)"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i+1}: {test_case['description']}")
        print(f"{'='*60}")
        
        prompt = test_case["prompt"]
        attributes = test_case["attributes"]
        
        print(f"\n📝 Prompt: {prompt}")
        
        # 生成Nemotron格式的prompt
        nemotron_prompt, attr_string = build_nemotron_prompt(prompt, attributes)
        
        print(f"\n📝 Generated Nemotron prompt:")
        print(f"```\n{nemotron_prompt}\n```")
        
        # 生成响应
        print("⚡ Generating response...")
        start_time = time.time()
        
        response = generate_with_nemotron(
            model=model,
            prompt_text=nemotron_prompt,
            max_tokens=512,
            temperature=0.7
        )
        
        end_time = time.time()
        
        # 清理输出
        if response.startswith(nemotron_prompt):
            response = response[len(nemotron_prompt):].strip()
        response = response.split("<extra_id_1>")[0].strip()
        
        print(f"\n🎯 Generated Response (took {end_time - start_time:.2f}s):")
        print(f"```\n{response}\n```")
        
        print(f"\n📊 Attribute string used: {attr_string}")
    
    print(f"\n✅ Nemotron-3-8B-SteerLM test completed!")
    print(f"🔧 Model: nvidia/nemotron-3-8b-chat-4k-steerlm")
    print(f"📚 Framework: NVIDIA NeMo")

if __name__ == "__main__":
    test_nemotron_model() 