#!/usr/bin/env python3
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
        print(f"```")
        print(f"{nemotron_prompt}")
        print(f"```")
        
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
