# dpo_rps_perturbation_final.py
# 最终版本：基于DPO模型使用RPS方法，在UltraFeedback前2000个prompt上测试

import os
import numpy as np
import pandas as pd
import json
import random
import time
import openai
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
from collections import Counter

# 🇨🇳 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'

# 🤖 设置 OpenAI API
os.environ["OPENAI_API_KEY"] = "sk-XGGe5y0ZvLcQVFp6XnRizs7q47gsVnAbZx0Xr2mfcVlbr99f"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = "https://api2.aigcbest.top/v1"

print("✅ OpenAI 设置完成")

# 定义DPO的主方向向量
DPO_PREFERENCE_DIRECTIONS = {
    "v3": {"vector": (0.9848, 0.1736), "angle": 10},
    "v4": {"vector": (0.9659, 0.2588), "angle": 15}, 
    "v5": {"vector": (0.9397, 0.3420), "angle": 20},
    "v6": {"vector": (0.9063, 0.4226), "angle": 25},
    "v7": {"vector": (0.8660, 0.5000), "angle": 30},
    "v8": {"vector": (0.8192, 0.5736), "angle": 35},
    "v9": {"vector": (0.7660, 0.6428), "angle": 40},
    "v10": {"vector": (0.7071, 0.7071), "angle": 45},
}

def setup_environment():
    """设置环境和随机种子"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return device

def get_dpo_angle_perturbations(v_main, angle_range=(-30, 30), step=5, theta_max=25, top_k=5):
    """为DPO方向生成角度扰动（参考angle_based.py）"""
    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    main_angle = np.degrees(np.arctan2(v_main[1], v_main[0]))
    
    angle_offsets = np.arange(angle_range[0], angle_range[1] + 1, step)
    perturbed_vs = []
    perturbed_angles = []
    angle_diffs = []

    for offset in angle_offsets:
        new_angle = main_angle + offset
        angle_rad = np.radians(new_angle)
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        angle_diff = angle_between(v, v_main)
        if angle_diff <= theta_max:
            perturbed_vs.append(v)
            perturbed_angles.append(new_angle)
            angle_diffs.append(angle_diff)

    sorted_indices = np.argsort(angle_diffs)
    top_indices = sorted_indices[:top_k]
    
    valid_vs = [perturbed_vs[i] for i in top_indices]
    valid_angles = [perturbed_angles[i] for i in top_indices]
    
    print(f"✅ 生成了 {len(valid_vs)} 个有效扰动方向")
    for i, (v, a) in enumerate(zip(valid_vs, valid_angles)):
        print(f"  扰动{i+1}: angle={a:.1f}°, v=({v[0]:.4f}, {v[1]:.4f})")
    
    return valid_vs, valid_angles

def load_dpo_models(device):
    """加载DPO模型和Reward模型"""
    print("🤖 Loading DPO model...")
    try:
        dpo_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # 🔄 替换为真正的DPO模型
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        ).to(device)
        
        dpo_tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta",  # 🔄 替换为对应的tokenizer
            trust_remote_code=True
        )
        dpo_tokenizer.padding_side = "left"
        if dpo_tokenizer.pad_token_id is None:
            dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
        print("✅ DPO model loaded!")
        
    except Exception as e:
        print(f"❌ DPO model loading failed: {e}")
        return None, None, None, None
    
    print("🏆 Loading Reward model...")
    try:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True
        ).to(device)
        
        reward_tokenizer = AutoTokenizer.from_pretrained(
            "Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1",
            trust_remote_code=True
        )
        print("✅ Reward model loaded!")
        
    except Exception as e:
        print(f"❌ Reward model loading failed: {e}")
        return None, None, None, None
    
    return dpo_model, dpo_tokenizer, reward_model, reward_tokenizer

def build_dpo_input(prompt, v1, v2):
    """构造DPO输入（使用DPA格式保持一致）"""
    h = int(np.round(v1 * 100))
    v = int(np.round(v2 * 100))
    sys_instruction = f"You are a helpful assistant. Your response should maximize weighted rating = helpfulness*{h} + verbosity*{v}."
    return [{"role": "user", "content": f"{sys_instruction}\n\n{prompt}"}]

def generate_dpo_responses_for_perturbation(prompt, prompt_id, v_vec, angle_deg, 
                                          dpo_model, dpo_tokenizer, device, num_responses=3):
    """为单个扰动方向生成DPO响应"""
    try:
        v1, v2 = v_vec[0], v_vec[1]
        messages = build_dpo_input(prompt, v1, v2)
        input_ids = dpo_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        
        max_input_len = input_ids.shape[1]
        max_new_tokens = min(2048, 4096 - max_input_len)
        
        with torch.no_grad():
            outputs = dpo_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=num_responses,
                pad_token_id=dpo_tokenizer.eos_token_id,
                top_p=0.9,  # 🔄 添加top_p参数
                repetition_penalty=1.1  # 🔄 添加repetition_penalty参数
            )
        
        responses = []
        for i in range(num_responses):
            generated_tokens = outputs[i][input_ids.shape[1]:]
            decoded = dpo_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "perturbation_vector": f"({v1:.4f}, {v2:.4f})",
                "perturbation_angle": angle_deg,
                "response_id": i + 1,
                "response": decoded
            })
        
        return responses
    
    except Exception as e:
        print(f"⚠️ Error generating for prompt {prompt_id}, angle {angle_deg}: {e}")
        return []

def score_dpo_response(prompt, response, reward_model, reward_tokenizer, device, v1, v2):
    """使用Reward Model为DPO响应打分"""
    try:
        template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        inputs = reward_tokenizer(
            template.format(prompt=prompt, response=response),
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = reward_model(**inputs).logits.squeeze().cpu().numpy()
        
        helpfulness = logits[9]
        verbosity = logits[4]
        rps_score = v1 * helpfulness + v2 * verbosity
        
        return {
            "helpfulness": float(helpfulness),
            "verbosity": float(verbosity),
            "v1": float(v1),
            "v2": float(v2),
            "rps_score": float(rps_score)
        }
    
    except Exception as e:
        print(f"⚠️ Error scoring response: {e}")
        return {
            "helpfulness": 0.0,
            "verbosity": 0.0,
            "v1": float(v1),
            "v2": float(v2),
            "rps_score": 0.0
        }

def run_dpo_rps_for_direction(direction_name, direction_info, prompts, prompt_ids,
                             dpo_model, dpo_tokenizer, reward_model, reward_tokenizer,
                             device, output_dir, batch_size=8):  # 🔄 从4改为8
    """为单个主方向运行DPO RPS实验，支持断点续跑"""
    
    print(f"\n🎯 处理方向 {direction_name}: {direction_info['vector']} ({direction_info['angle']}°)")
    
    # 检查是否已有最终结果
    output_file = os.path.join(output_dir, f"dpo_rps_{direction_name}_best_responses.csv")
    if os.path.exists(output_file):
        print(f"✅ 发现已有最终结果，直接加载: {output_file}")
        df = pd.read_csv(output_file)
        return df.to_dict("records")
    
    # 生成扰动方向
    main_v = np.array(direction_info["vector"])
    valid_vs, valid_angles = get_dpo_angle_perturbations(
        v_main=main_v,
        angle_range=(-30, 30),
        step=5,
        theta_max=25,
        top_k=5
    )
    
    # 为每个扰动方向生成响应
    all_perturbation_results = []
    
    for i, (v_vec, angle_deg) in enumerate(zip(valid_vs, valid_angles)):
        print(f"\n📐 扰动方向 {i+1}: angle={angle_deg:.1f}°, v=({v_vec[0]:.4f}, {v_vec[1]:.4f})")
        
        # 检查单个扰动方向的断点续跑
        perturbation_file = os.path.join(output_dir, f"temp_{direction_name}_perturbation_{i+1}.csv")
        if os.path.exists(perturbation_file):
            print(f"🔁 加载已有扰动结果: {perturbation_file}")
            df_temp = pd.read_csv(perturbation_file)
            all_perturbation_results.extend(df_temp.to_dict("records"))
            continue
        
        v1, v2 = v_vec[0], v_vec[1]
        perturbation_results = []
        
        # 批量处理prompts
        for start in tqdm(range(0, len(prompts), batch_size), 
                         desc=f"DPO RPS {direction_name} 扰动{i+1}"):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_ids = prompt_ids[start:end]
            
            for j, (prompt, pid) in enumerate(zip(batch_prompts, batch_ids)):
                # 生成响应
                responses = generate_dpo_responses_for_perturbation(
                    prompt, pid, v_vec, angle_deg,
                    dpo_model, dpo_tokenizer, device, num_responses=3
                )
                
                if not responses:
                    continue
                
                # 为每个响应打分
                scored_responses = []
                for resp_data in responses:
                    scores = score_dpo_response(
                        resp_data["prompt"], 
                        resp_data["response"],
                        reward_model, reward_tokenizer, device,
                        v1, v2
                    )
                    resp_data.update(scores)
                    scored_responses.append(resp_data)
                
                # 选择最佳响应
                if scored_responses:
                    best_response = max(scored_responses, key=lambda x: x["rps_score"])
                    best_response["is_best"] = True
                    best_response["all_scores"] = [r["rps_score"] for r in scored_responses]
                    perturbation_results.append(best_response)
        
        # 保存单个扰动的结果
        if perturbation_results:
            df_temp = pd.DataFrame(perturbation_results)
            df_temp.to_csv(perturbation_file, index=False)
            print(f"💾 保存扰动{i+1}结果: {len(perturbation_results)} 个响应")
        
        all_perturbation_results.extend(perturbation_results)
    
    # 为每个prompt选择所有扰动中的最佳响应
    prompt_best_results = {}
    for result in all_perturbation_results:
        pid = result["prompt_id"]
        if pid not in prompt_best_results or result["rps_score"] > prompt_best_results[pid]["rps_score"]:
            prompt_best_results[pid] = result
    
    final_results = list(prompt_best_results.values())
    
    # 保存最终结果
    df = pd.DataFrame(final_results)
    df.to_csv(output_file, index=False)
    
    # 清理临时文件
    for i in range(len(valid_vs)):
        temp_file = os.path.join(output_dir, f"temp_{direction_name}_perturbation_{i+1}.csv")
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"✅ {direction_name} 完成，最佳响应保存至: {output_file}")
    print(f"📊 处理了 {len(final_results)} 个prompts，平均RPS得分: {df['rps_score'].mean():.2f}")
    
    return final_results

def load_dpo_baseline_results(dpo_outputs_dir):
    """加载已有的DPO baseline结果"""
    baseline_results = {}
    
    for direction_name in DPO_PREFERENCE_DIRECTIONS.keys():
        file_path = os.path.join(dpo_outputs_dir, f"dpo_responses_{direction_name}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            baseline_results[direction_name] = df
            print(f"✅ 加载 {direction_name} baseline: {len(df)} 响应")
        else:
            print(f"⚠️ 未找到 {direction_name} baseline 文件: {file_path}")
    
    return baseline_results

def run_gpt_judging(input_path, model="gpt-4o-mini", sleep_time=1.0, max_retries=3):
    """使用 OpenAI GPT 模型进行自动评估，支持断点续跑"""
    output_path = input_path.replace("_judge_input.jsonl", "_results.jsonl")
    
    # 断点续跑检查
    completed_ids = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                results.append(item)
                completed_ids.add(item["pair_id"])
        print(f"🔁 已加载 {len(completed_ids)} 条历史结果")

    # 加载输入数据
    with open(input_path, "r", encoding="utf-8") as f:
        all_prompts = [json.loads(line) for line in f]

    start_time = time.time()

    # 遍历数据评估
    for item in tqdm(all_prompts, desc=f"🧠 {model} 评估中"):
        pid = item["pair_id"]
        if pid in completed_ids:
            continue

        if item.get("auto_result") == "Tie":
            item["gpt_judgment"] = "Tie"
            print(f"🤝 pair_id={pid} → Auto-Tie")
            results.append(item)
            # 实时保存
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            continue

        prompt = item["formatted_prompt"]
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                reply = response.choices[0].message["content"].strip()
                item["gpt_raw_response"] = reply

                # 解析判断结果
                last_line = reply.strip().splitlines()[-1].strip().upper()
                if "MORE HELPFUL: A" in last_line or last_line == "A":
                    item["gpt_judgment"] = "A"
                elif "MORE HELPFUL: B" in last_line or last_line == "B":
                    item["gpt_judgment"] = "B"
                else:
                    item["gpt_judgment"] = "Unclear"

                print(f"✅ pair_id={pid} → {item['gpt_judgment']}")
                break
            except Exception as e:
                item["error"] = str(e)
                print(f"❌ pair_id={pid} → Error: {str(e)}")
                time.sleep(sleep_time)
        else:
            item["gpt_judgment"] = "Error"
            print(f"❌ pair_id={pid} → Failed after {max_retries} attempts")

        results.append(item)
        time.sleep(sleep_time)

        # 实时保存（每10个保存一次）
        if len(results) % 10 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 最终保存
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    end_time = time.time()
    print(f"\n✅ 评估完成，共计 {len(results)} 条")
    print(f"🕒 耗时：{end_time - start_time:.1f} 秒")
    print(f"📁 输出结果路径：{output_path}")
    
    return output_path

def analyze_gpt_judgment_results(results_path):
    """分析 GPT 评审结果，输出类似compare_result的统计"""
    with open(results_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    def judge_winner(item):
        judgment = item.get("gpt_judgment", "")
        a_origin = item.get("a_origin")
        b_origin = item.get("b_origin")

        if judgment == "Tie":
            return "Tie"
        elif judgment == "A":
            return a_origin
        elif judgment == "B":
            return b_origin
        elif judgment == "Unclear":
            return "Unclear"
        elif judgment == "Error":
            return "Error"
        else:
            return "Invalid"

    outcomes = [judge_winner(item) for item in data]
    counter = Counter(outcomes)

    total = sum(counter.values())
    win_rps = counter.get("RPS", 0)
    win_baseline = counter.get("Baseline", 0)
    tie = counter.get("Tie", 0)
    unclear = counter.get("Unclear", 0)
    error = counter.get("Error", 0)

    print(f"\n📊 GPT-4 判断结果统计 ({results_path.split('/')[-1]}):")
    print(f"总计: {total}")
    print(f"RPS 获胜: {win_rps} ({win_rps/total*100:.1f}%)")
    print(f"Baseline 获胜: {win_baseline} ({win_baseline/total*100:.1f}%)")
    print(f"平局: {tie} ({tie/total*100:.1f}%)")
    print(f"不清楚: {unclear} ({unclear/total*100:.1f}%)")
    print(f"错误: {error} ({error/total*100:.1f}%)")
    
    if win_rps + win_baseline > 0:
        rps_win_rate = win_rps / (win_rps + win_baseline) * 100
        print(f"\n🏆 RPS vs Baseline 胜率: {rps_win_rate:.1f}%")
    
    return counter

def merge_and_compare_with_gpt(rps_results, baseline_results, direction_name, output_dir, 
                              run_gpt_judge=True):
    """合并RPS结果和baseline结果，并进行GPT判断"""
    
    # 将RPS结果转为DataFrame
    df_rps = pd.DataFrame(rps_results)
    df_rps = df_rps.rename(columns={"response": "rps_best_response"})
    
    # 处理baseline结果
    if direction_name in baseline_results:
        df_baseline = baseline_results[direction_name].copy()
        
        # 为baseline计算得分（如果还没有的话）
        if "dpa_score" not in df_baseline.columns:
            print(f"⚠️ {direction_name} baseline缺少评分，跳过比较")
            return None
        
        # 选择每个prompt的最佳baseline响应
        df_baseline_best = df_baseline.loc[df_baseline.groupby("prompt_id")["dpa_score"].idxmax()]
        df_baseline_best = df_baseline_best.rename(columns={
            "response": "baseline_best_response",
            "dpa_score": "baseline_score"
        })
        
        # 合并数据
        df_merged = pd.merge(
            df_rps[["prompt_id", "prompt", "rps_best_response", "rps_score"]],
            df_baseline_best[["prompt_id", "baseline_best_response", "baseline_score"]],
            on="prompt_id",
            how="inner"
        )
        
        # 保存合并结果
        comparison_file = os.path.join(output_dir, f"{direction_name}_rps_vs_baseline_comparison.csv")
        df_merged.to_csv(comparison_file, index=False)
        
        print(f"✅ {direction_name} 比较结果保存至: {comparison_file}")
        print(f"📊 RPS平均得分: {df_merged['rps_score'].mean():.2f}")
        print(f"📊 Baseline平均得分: {df_merged['baseline_score'].mean():.2f}")
        
        # 计算基于得分的胜率
        rps_wins = (df_merged['rps_score'] > df_merged['baseline_score']).sum()
        total = len(df_merged)
        win_rate = rps_wins / total * 100
        print(f"📈 基于得分的RPS胜率: {win_rate:.1f}% ({rps_wins}/{total})")
        
        if run_gpt_judge:
            # 生成GPT判断输入（使用与compare_result相同的格式）
            judge_input_file = generate_gpt_judge_input(df_merged, direction_name, output_dir)
            
            # 运行GPT判断
            print(f"\n🧠 开始GPT-4判断 {direction_name}...")
            gpt_results_file = run_gpt_judging(judge_input_file, model="gpt-4o-mini")
            
            # 分析GPT判断结果
            print(f"\n📈 分析GPT判断结果 {direction_name}:")
            gpt_counter = analyze_gpt_judgment_results(gpt_results_file)
            
            return {
                "comparison_df": df_merged,
                "gpt_results_file": gpt_results_file,
                "gpt_counter": gpt_counter
            }
        
        return {"comparison_df": df_merged}
    
    else:
        print(f"⚠️ 未找到 {direction_name} 的baseline结果")
        return None

def generate_gpt_judge_input(comparison_df, direction_name, output_dir):
    """生成GPT-4判断的输入文件，格式与compare_result一致"""
    judge_data = []
    
    for idx, row in comparison_df.iterrows():
        prompt = row["prompt"]
        rps_response = str(row["rps_best_response"]) if pd.notna(row["rps_best_response"]) else ""
        baseline_response = str(row["baseline_best_response"]) if pd.notna(row["baseline_best_response"]) else ""
        
        if rps_response.strip() == baseline_response.strip():
            judge_data.append({
                "pair_id": idx,
                "prompt_id": row["prompt_id"],
                "auto_result": "Tie",
                "baseline_id": None,
                "a_origin": "RPS",
                "b_origin": "Baseline",
                "formatted_prompt": "[Same responses, auto tie]"
            })
            continue
        
        # 随机A/B顺序
        if random.random() < 0.5:
            response_a, response_b = rps_response, baseline_response
            a_origin, b_origin = "RPS", "Baseline"
        else:
            response_a, response_b = baseline_response, rps_response
            a_origin, b_origin = "Baseline", "RPS"
        
        formatted_prompt = f"""[HH-RLHF]: For the following query to a chatbot, which response is more helpful?

Query: {prompt}

Response A: {response_a}

Response B: {response_b}

FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful.
SECOND, on a new line, state only 'A' or 'B' to indicate which response is more helpful.

Format:
Comparison: ...
More helpful: A/B"""
        
        judge_data.append({
            "pair_id": idx,
            "prompt_id": row["prompt_id"],
            "auto_result": None,
            "baseline_id": None,
            "a_origin": a_origin,
            "b_origin": b_origin,
            "formatted_prompt": formatted_prompt
        })
    
    # 保存JSONL文件（文件名格式与compare_result一致）
    output_file = os.path.join(output_dir, f"{direction_name}_pairwise_randomized_rps_judge_input.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in judge_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ GPT判断输入文件保存至: {output_file}")
    return output_file

def main():
    """主函数"""
    # 设置环境
    device = setup_environment()
    
    # 🔄 修复路径，指向正确的baseline结果目录
    dpo_outputs_dir = "/root/rps/data/dpo_baseline_outputs"  # 修复路径
    rps_output_dir = "/root/rps/data/dpo_rps_results"
    comparison_output_dir = "/root/rps/data/dpo_rps_comparisons"
    
    for dir_path in [rps_output_dir, comparison_output_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 🔄 改为2000个prompts，完整实验
    print("📦 Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    prompts = ds["prompt"][:2000]  # 🔄 从100改为2000
    prompt_ids = list(range(len(prompts)))
    print(f"✅ 加载了 {len(prompts)} 个prompts")
    
    # 加载模型
    dpo_model, dpo_tokenizer, reward_model, reward_tokenizer = load_dpo_models(device)
    if dpo_model is None:
        print("❌ 模型加载失败，退出")
        return
    
    # 加载DPO baseline结果
    baseline_results = load_dpo_baseline_results(dpo_outputs_dir)
    
    # 为每个方向运行RPS实验
    # 自动包含所有定义的方向
    directions_to_test = list(DPO_PREFERENCE_DIRECTIONS.keys())
    
    all_results = {}
    
    print(f"\n🚀 开始在前{len(prompts)}个prompts上测试DPO RPS方法")
    print(f"📐 将测试方向: {directions_to_test}")
    
    for direction_name in directions_to_test:
        if direction_name not in DPO_PREFERENCE_DIRECTIONS:
            print(f"⚠️ 跳过未定义的方向: {direction_name}")
            continue
            
        direction_info = DPO_PREFERENCE_DIRECTIONS[direction_name]
        
        print(f"\n🚀 开始处理方向 {direction_name}")
        
        # 运行RPS实验（支持断点续跑）
        rps_results = run_dpo_rps_for_direction(
            direction_name, direction_info,
            prompts, prompt_ids,
            dpo_model, dpo_tokenizer, reward_model, reward_tokenizer,
            device, rps_output_dir,
            batch_size=8  # 🔄 从4改为8
        )
        
        # 与baseline比较并进行GPT判断
        comparison_results = merge_and_compare_with_gpt(
            rps_results, baseline_results, direction_name, comparison_output_dir,
            run_gpt_judge=True
        )
        
        all_results[direction_name] = comparison_results
    
    # 汇总所有结果
    print("\n🎉 所有实验完成！")
    print(f"📁 RPS结果保存在: {rps_output_dir}")
    print(f"📁 比较结果保存在: {comparison_output_dir}")
    
    print("\n📊 总体结果汇总:")
    for direction_name, results in all_results.items():
        if results and "gpt_counter" in results:
            counter = results["gpt_counter"]
            total = sum(counter.values())
            win_rps = counter.get("RPS", 0)
            win_baseline = counter.get("Baseline", 0)
            if win_rps + win_baseline > 0:
                rps_win_rate = win_rps / (win_rps + win_baseline) * 100
                print(f"  {direction_name}: RPS胜率 {rps_win_rate:.1f}% ({win_rps}/{win_rps + win_baseline})")

    # 输出结果文件格式说明
    print(f"\n📄 生成的结果文件格式:")
    print(f"  - 类似你的compare_result文件夹中的格式")
    print(f"  - 文件名: {direction_name}_pairwise_randomized_rps_results.jsonl")
    print(f"  - 包含GPT-4判断结果，可直接用于分析")

if __name__ == "__main__":
    main()