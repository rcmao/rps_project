# ================== Colab 依赖安装 ==================
# 在Colab中可直接运行此脚本
!pip install -q transformers accelerate datasets torch tqdm

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ================== SteerLM Prompt 构建 ==================
def build_steerlm_prompt(prompt, attr_dict=None):
    # 默认属性全4（可自定义）
    if attr_dict is None:
        attr_dict = {
            "quality": 4, "understanding": 4, "correctness": 4, "coherence": 4, "complexity": 4,
            "verbosity": 4, "toxicity": 0, "humor": 0, "creativity": 0, "violence": 0,
            "helpfulness": 4, "not_appropriate": 0, "hate_speech": 0, "sexual_content": 0,
            "fails_task": 0, "political_content": 0, "moral_judgement": 0, "lang": "en"
        }
    attr_string = ",".join([f"{k}:{v}" for k, v in attr_dict.items()])
    steerlm_prompt = f"""<extra_id_0>System
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
<extra_id_2>{attr_string}
"""
    return steerlm_prompt

# ================== 加载模型 ==================
def load_nemotron_model(model_name="nvidia/nemotron-3-8b-chat-4k-steerlm"):
    print(f"🤖 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False   # 关键参数！
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

# ================== 生成函数 ==================
def generate_with_nemotron(model, tokenizer, prompt_text, max_tokens=512, temperature=0.7):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
        )
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    # 提取 Assistant 部分
    if "<extra_id_1>Assistant" in generated_text:
        response = generated_text.split("<extra_id_1>Assistant")[-1]
        # 去掉属性行
        if "<extra_id_2>" in response:
            response = response.split("<extra_id_2>")[-1]
        response = response.strip()
    else:
        response = generated_text
    return response

# ================== 主流程 ==================
def main():
    print("🚀 Starting Nemotron-3-8B SteerLM test in Colab!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model, tokenizer = load_nemotron_model()
    # 加载数据
    print("📦 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:10]  # 只取前10个

    # 批量生成前10个prompt的响应
    print("\n⚡ Generating responses for first 10 prompts...")
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        steerlm_prompt = build_steerlm_prompt(prompt)
        response = generate_with_nemotron(
            model=model,
            tokenizer=tokenizer,
            prompt_text=steerlm_prompt,
            max_tokens=512,
            temperature=0.7
        )
        results.append({"prompt_id": i, "prompt": prompt, "response": response})
        print(f"\nPrompt {i+1}:\n{prompt}\n---\nResponse:\n{response}\n{'='*40}")

    print("\n✅ All done! You can now analyze the results.")

if __name__ == "__main__":
    main() 