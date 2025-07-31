#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO RPS 打分和选择最佳响应脚本 - 服务器版本
对每个方向的生成结果进行打分，并选出每个prompt的最佳响应
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import json
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# 设置内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_reward_scoring(input_dir, output_dir, model_name="Haoxiang-Wang/RewardModel-Mistral-7B-for-DPA-v1"):
    """
    对输入目录中的所有CSV文件进行打分
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        model_name: 奖励模型名称
    """
    print(f"🤖 Loading reward model: {model_name}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # 加载模型和tokenizer
    try:
        rm = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Reward model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load reward model: {e}")
        print("🔄 Trying CPU fallback...")
        try:
            rm = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✅ Reward model loaded on CPU!")
        except Exception as e2:
            print(f"❌ CPU loading also failed: {e2}")
            return

    def score_response(prompt, response):
        """对单个响应进行打分"""
        try:
            template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
            inputs = tokenizer(
                template.format(prompt=prompt, response=response), 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
            
            with torch.no_grad():
                logits = rm(**inputs).logits.squeeze().cpu().numpy()
            
            # 返回helpfulness和verbosity分数
            return logits[9], logits[4]  # helpfulness, verbosity
            
        except Exception as e:
            print(f"⚠️ Scoring error: {e}")
            return None, None

    # 处理所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"📁 Found {len(csv_files)} CSV files to process")
    
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"\n📄 Processing: {file}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"   📊 Loaded {len(df)} rows")
            
            # 初始化分数列
            if "helpfulness" not in df.columns:
                df["helpfulness"] = np.nan
            if "verbosity" not in df.columns:
                df["verbosity"] = np.nan
            
            # 统计需要打分的行数
            need_scoring = df[df["helpfulness"].isna() | df["verbosity"].isna()].shape[0]
            print(f"   🎯 Need to score {need_scoring} responses")
            
            if need_scoring == 0:
                print(f"   ✅ All responses already scored, skipping...")
                continue
            
            # 逐行打分
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {file}"):
                if pd.notnull(row["helpfulness"]) and pd.notnull(row["verbosity"]):
                    continue
                
                try:
                    h, v = score_response(row["prompt"], row["response"])
                    df.loc[i, "helpfulness"] = h
                    df.loc[i, "verbosity"] = v
                    
                    # 每100行保存一次，防止数据丢失
                    if (i + 1) % 100 == 0:
                        temp_save_path = os.path.join(output_dir, file.replace(".csv", "_temp_scored.csv"))
                        df.to_csv(temp_save_path, index=False)
                        
                except Exception as e:
                    print(f"   ❌ Error on row {i}: {e}")
                    continue
            
            # 保存最终结果
            save_path = os.path.join(output_dir, file.replace(".csv", "_scored.csv"))
            df.to_csv(save_path, index=False)
            print(f"   💾 Scored file saved to: {save_path}")
            
            # 清理临时文件
            temp_file = os.path.join(output_dir, file.replace(".csv", "_temp_scored.csv"))
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"   ❌ Error processing file {file}: {e}")
            continue

def select_best_response(scored_dir, output_path):
    """
    从已打分的文件中选出每个prompt的最佳响应
    Args:
        scored_dir: 已打分的文件目录
        output_path: 输出文件路径
    """
    print(f"🔍 Selecting best responses from {scored_dir}")
    
    dfs = []
    scored_files = [f for f in os.listdir(scored_dir) if f.endswith("_scored.csv")]
    
    if not scored_files:
        print(f"❌ No scored files found in {scored_dir}")
        return
    
    print(f"📁 Found {len(scored_files)} scored files")
    
    for file in scored_files:
        try:
            file_path = os.path.join(scored_dir, file)
            df = pd.read_csv(file_path)
            print(f"   📄 Loaded {file}: {len(df)} rows")
            
            # 检查必要的列是否存在
            required_cols = ["prompt_id", "prompt", "response", "helpfulness", "verbosity"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ⚠️ Missing columns in {file}: {missing_cols}")
                continue
            
            # 计算总分
            if "main_v1" in df.columns and "main_v2" in df.columns:
                df["score_total"] = df["main_v1"] * df["helpfulness"] + df["main_v2"] * df["verbosity"]
            elif "v1_p" in df.columns and "v2_p" in df.columns:
                df["score_total"] = df["v1_p"] * df["helpfulness"] + df["v2_p"] * df["verbosity"]
            elif "v1" in df.columns and "v2" in df.columns:
                df["score_total"] = df["v1"] * df["helpfulness"] + df["v2"] * df["verbosity"]
            else:
                print(f"   ⚠️ No direction vector columns found in {file}, using default weights")
                # 使用默认权重
                df["score_total"] = 0.7071 * df["helpfulness"] + 0.7071 * df["verbosity"]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"   ❌ Error processing {file}: {e}")
            continue
    
    if not dfs:
        print("❌ No valid data found for best response selection")
        return
    
    # 合并所有数据
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total {len(df_all)} responses from all files")
    
    # 按prompt_id分组，选出得分最高的响应
    df_best = df_all.loc[df_all.groupby("prompt_id")["score_total"].idxmax()].copy()
    df_best = df_best.rename(columns={"response": "best_response"})
    
    # 保存结果
    df_best.to_csv(output_path, index=False)
    print(f"✅ Best responses saved to: {output_path}")
    print(f"�� Selected {len(df_best)} best responses from {len(df_all)} total responses")

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 设置路径
    base_dir = "/root/rps/data/dpo_rps_generated"
    output_dir = "/root/rps/data"
    
    # 检查输入目录是否存在
    if not os.path.exists(base_dir):
        print(f"❌ Input directory not found: {base_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 要处理的方向
    directions = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    
    print("🚀 Starting DPO RPS scoring and best response selection")
    print(f"�� Input directory: {base_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Directions to process: {directions}")
    
    start_time = time.time()
    
    for i, direction in enumerate(directions, 1):
        print(f"\n{'='*60}")
        print(f"📊 Processing direction {i}/{len(directions)}: {direction}")
        print(f"{'='*60}")
        
        input_dir = os.path.join(base_dir, direction)
        if not os.path.exists(input_dir):
            print(f"⚠️ Directory not found: {input_dir}, skipping...")
            continue
        
        scored_dir = os.path.join(input_dir, "scored")
        
        try:
            # Step 1: 打分
            print(f"🎯 Step 1: Scoring responses for {direction}")
            run_reward_scoring(input_dir, scored_dir)
            
            # Step 2: 选出best response
            print(f"🏆 Step 2: Selecting best responses for {direction}")
            output_path = os.path.join(output_dir, f"{direction}_best_response.csv")
            select_best_response(scored_dir, output_path)
            
            print(f"✅ Completed processing {direction}")
            
        except Exception as e:
            print(f"❌ Error processing {direction}: {e}")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🎉 All processing completed!")
    print(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📂 Results saved to: {output_dir}")
    
    # 列出生成的文件
    output_files = [f for f in os.listdir(output_dir) if f.endswith("_best_response.csv")]
    print(f"📄 Generated {len(output_files)} best response files:")
    for file in sorted(output_files):
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   - {file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()