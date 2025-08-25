 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO RPS vs Baseline DPO 比较脚本
生成pairwise比较数据，用于GPT-4o-mini评判
"""

import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm.auto import tqdm

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)

def generate_pairwise_jsonl_for_gpt_judging(
    baseline_path, 
    rps_path, 
    output_path, 
    direction_name,
    seed=42,
    max_pairs=None  # 改为None，不限制数量
):
    """
    生成用于GPT评判的pairwise比较数据
    Args:
        baseline_path: baseline DPO文件路径
        rps_path: RPS best response文件路径  
        output_path: 输出jsonl文件路径
        direction_name: 方向名称（如v3, v4等）
        seed: 随机种子
        max_pairs: 最大比较对数，None表示不限制
    """
    print(f"🔄 Generating pairwise comparisons for {direction_name}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 读取数据
    try:
        baseline_df = pd.read_csv(baseline_path)
        rps_df = pd.read_csv(rps_path)
        print(f" Loaded baseline: {len(baseline_df)} rows")
        print(f"📊 Loaded RPS {direction_name}: {len(rps_df)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 确保列名一致
    baseline_response_col = 'response' if 'response' in baseline_df.columns else 'best_response'
    rps_response_col = 'best_response' if 'best_response' in rps_df.columns else 'response'
    
    # 合并数据，确保prompt_id匹配
    merged_df = pd.merge(
        baseline_df[['prompt_id', 'prompt', baseline_response_col]], 
        rps_df[['prompt_id', 'prompt', rps_response_col]], 
        on='prompt_id', 
        suffixes=('_baseline', '_rps')
    )
    
    print(f" Merged data: {len(merged_df)} matching prompts")
    
    if len(merged_df) == 0:
        print("❌ No matching prompts found")
        return
    
    # 限制比较对数（如果指定了max_pairs）
    if max_pairs is not None and len(merged_df) > max_pairs:
        merged_df = merged_df.sample(n=max_pairs, random_state=seed)
        print(f" Sampled {max_pairs} pairs for comparison")
    else:
        print(f"📊 Using all {len(merged_df)} pairs for comparison")
    
    # 生成pairwise比较数据
    pairwise_data = []
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Generating pairs"):
        prompt = row['prompt_baseline']  # 两个prompt应该相同
        
        # 随机决定A和B的位置
        if random.random() < 0.5:
            response_a = row[baseline_response_col]
            response_b = row[rps_response_col]
            a_is_baseline = True
        else:
            response_a = row[rps_response_col]
            response_b = row[baseline_response_col]
            a_is_baseline = False
        
        # 构建比较数据
        comparison_item = {
            "id": f"{direction_name}_{idx}",
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "a_is_baseline": a_is_baseline,
            "direction": direction_name,
            "prompt_id": row['prompt_id']
        }
        
        pairwise_data.append(comparison_item)
    
    # 保存为jsonl格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in pairwise_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Generated {len(pairwise_data)} pairwise comparisons")
    print(f"💾 Saved to: {output_path}")
    
    # 统计A/B位置分布
    baseline_as_a = sum(1 for item in pairwise_data if item['a_is_baseline'])
    rps_as_a = len(pairwise_data) - baseline_as_a
    print(f"📈 Position distribution: Baseline as A: {baseline_as_a}, RPS as A: {rps_as_a}")

def run_gpt_judging(input_path, output_path, model="gpt-4o-mini", sleep_time=1.0, max_retries=3):
    """
    使用GPT-4o-mini进行评判
    Args:
        input_path: 输入的jsonl文件路径
        output_path: 输出的评判结果文件路径
        model: 使用的模型
        sleep_time: 请求间隔时间
        max_retries: 最大重试次数
    """
    import openai
    
    # 设置OpenAI API
    os.environ["OPENAI_API_KEY"] = "sk-XGGe5y0ZvLcQVFp6XnRizs7q47gsVnAbZx0Xr2mfcVlbr99f"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = "https://api2.aigcbest.top/v1"
    
    print(f"🤖 Starting GPT judging with {model}")
    
    # 读取输入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"�� Loaded {len(data)} comparisons to judge")
    
    # 评判提示模板
    judge_prompt_template = """You are an expert evaluator. Please compare two AI assistant responses to the same user prompt and determine which one is better.

User Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Please evaluate based on:
1. Helpfulness: How well does the response address the user's question or request?
2. Accuracy: Is the information provided correct and reliable?
3. Clarity: Is the response clear and easy to understand?
4. Completeness: Does the response provide a comprehensive answer?

Please respond with ONLY one of the following:
- "A" if Response A is better
- "B" if Response B is better  
- "TIE" if they are equally good

Your choice:"""

    results = []
    
    for i, item in tqdm(enumerate(data), total=len(data), desc="Judging"):
        prompt = judge_prompt_template.format(
            prompt=item['prompt'],
            response_a=item['response_a'],
            response_b=item['response_b']
        )
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                
                judgment = response.choices[0].message.content.strip().upper()
                
                # 验证判断结果
                if judgment in ['A', 'B', 'TIE']:
                    break
                else:
                    print(f"⚠️ Invalid judgment '{judgment}' for item {i}, retrying...")
                    judgment = 'TIE'  # 默认平局
                    
            except Exception as e:
                print(f"❌ Error judging item {i} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    judgment = 'TIE'  # 最后一次尝试失败，设为平局
                else:
                    time.sleep(sleep_time * 2)  # 失败时等待更长时间
                    continue
        
        # 记录结果
        result_item = item.copy()
        result_item['judgment'] = judgment
        result_item['model'] = model
        result_item['judgment_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        results.append(result_item)
        
        # 保存中间结果
        if (i + 1) % 50 == 0:
            temp_output_path = output_path.replace('.jsonl', f'_temp_{i+1}.jsonl')
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"�� Saved intermediate results: {temp_output_path}")
        
        time.sleep(sleep_time)
    
    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Judging completed! Results saved to: {output_path}")
    
    # 统计结果
    judgments = [r['judgment'] for r in results]
    a_wins = judgments.count('A')
    b_wins = judgments.count('B')
    ties = judgments.count('TIE')
    
    print(f"📊 Judgment statistics:")
    print(f"   A wins: {a_wins} ({a_wins/len(judgments)*100:.1f}%)")
    print(f"   B wins: {b_wins} ({b_wins/len(judgments)*100:.1f}%)")
    print(f"   Ties: {ties} ({ties/len(judgments)*100:.1f}%)")

def analyze_gpt_judgment_results(results_path, direction_name):
    """
    分析GPT评判结果
    Args:
        results_path: 评判结果文件路径
        direction_name: 方向名称
    """
    print(f"📊 Analyzing results for {direction_name}")
    
    # 读取结果
    with open(results_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
    
    # 统计结果
    baseline_wins = 0
    rps_wins = 0
    ties = 0
    
    for result in results:
        judgment = result['judgment']
        a_is_baseline = result['a_is_baseline']
        
        if judgment == 'A':
            if a_is_baseline:
                baseline_wins += 1
            else:
                rps_wins += 1
        elif judgment == 'B':
            if a_is_baseline:
                rps_wins += 1
            else:
                baseline_wins += 1
        else:  # TIE
            ties += 1
    
    total = len(results)
    
    print(f"📈 Analysis for {direction_name}:")
    print(f"   Total comparisons: {total}")
    print(f"   Baseline wins: {baseline_wins} ({baseline_wins/total*100:.1f}%)")
    print(f"   RPS wins: {rps_wins} ({rps_wins/total*100:.1f}%)")
    print(f"   Ties: {ties} ({ties/total*100:.1f}%)")
    
    # 计算胜率
    rps_win_rate = rps_wins / (baseline_wins + rps_wins) if (baseline_wins + rps_wins) > 0 else 0.5
    print(f"   RPS win rate (excluding ties): {rps_win_rate:.3f}")
    
    return {
        'direction': direction_name,
        'total': total,
        'baseline_wins': baseline_wins,
        'rps_wins': rps_wins,
        'ties': ties,
        'rps_win_rate': rps_win_rate
    }

def main():
    """主函数"""
    # 设置路径
    baseline_dir = "/root/rps/data/dpo_merged"
    rps_dir = "/root/rps/data/dpo_rps_best_resposne"
    output_dir = "/root/rps/data/dpo_compare_result"  # 修正输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(42)
    
    # 要处理的方向
    directions = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    
    print("🚀 Starting DPO RPS vs Baseline comparison")
    print(f" Baseline directory: {baseline_dir}")
    print(f"📂 RPS directory: {rps_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    all_analysis_results = []
    
    for direction in directions:
        print(f"\n{'='*60}")
        print(f"🎯 Processing direction: {direction}")
        print(f"{'='*60}")
        
        # 文件路径
        baseline_path = os.path.join(baseline_dir, f"dpo_responses_{direction}_merged.csv")
        rps_path = os.path.join(rps_dir, f"{direction}_best_response.csv")
        
        # 检查文件是否存在
        if not os.path.exists(baseline_path):
            print(f"❌ Baseline file not found: {baseline_path}")
            continue
        if not os.path.exists(rps_path):
            print(f"❌ RPS file not found: {rps_path}")
            continue
        
        # 生成pairwise比较数据 - 不限制数量
        pairwise_output = os.path.join(output_dir, f"{direction}_pairwise_randomized_angle_results.jsonl")
        generate_pairwise_jsonl_for_gpt_judging(
            baseline_path=baseline_path,
            rps_path=rps_path,
            output_path=pairwise_output,
            direction_name=direction,
            seed=42,
            max_pairs=None  # 改为None，使用所有2000对
        )
        
        # 使用GPT-4o-mini进行评判
        judged_output = os.path.join(output_dir, f"{direction}_judged_results.jsonl")
        run_gpt_judging(
            input_path=pairwise_output,
            output_path=judged_output,
            model="gpt-4o-mini",
            sleep_time=1.0,
            max_retries=3
        )
        
        # 分析结果
        analysis_result = analyze_gpt_judgment_results(judged_output, direction)
        all_analysis_results.append(analysis_result)
    
    # 保存汇总分析结果
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df = pd.DataFrame(all_analysis_results)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 Summary analysis saved to: {summary_path}")
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print(" OVERALL SUMMARY")
    print(f"{'='*60}")
    for result in all_analysis_results:
        print(f"{result['direction']}: RPS win rate = {result['rps_win_rate']:.3f} "
              f"({result['rps_wins']}/{result['baseline_wins'] + result['rps_wins']})")
    
    print(f"\n🎉 All comparisons completed!")

if __name__ == "__main__":
    main()