#!/usr/bin/env python3
"""
数据采样脚本：将每个prompt的5条响应采样到3条
处理所有 response_* 目录下的CSV文件，保持相同的目录结构
"""

import os
import pandas as pd
import random
from pathlib import Path
import shutil
from typing import List, Dict
import argparse

def setup_random_seed(seed: int = 42):
    """设置随机种子确保结果可复现"""
    random.seed(seed)

def find_response_directories(results_dir: str) -> List[str]:
    """找到所有 response_* 目录"""
    results_path = Path(results_dir)
    response_dirs = []
    
    for item in results_path.iterdir():
        if item.is_dir() and item.name.startswith('response_'):
            response_dirs.append(str(item))
    
    return sorted(response_dirs)

def process_csv_file(input_file: str, output_file: str):
    """处理单个CSV文件，将5条响应采样到3条"""
    print(f"处理文件: {input_file}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"读取文件失败 {input_file}: {e}")
        return False
    
    print(f"原始数据行数: {len(df)}")
    
    # 按prompt_id分组
    grouped = df.groupby('prompt_id')
    sampled_rows = []
    
    for prompt_id, group in grouped:
        # 检查这个prompt_id有多少个响应
        num_responses = len(group)
        
        if num_responses <= 3:
            # 如果原本就<=3条，保留所有
            sampled_rows.append(group)
        else:
            # 随机采样3条
            sampled_group = group.sample(n=3, random_state=42)
            sampled_rows.append(sampled_group)
    
    # 合并所有采样后的数据
    result_df = pd.concat(sampled_rows, ignore_index=True)
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"采样后数据行数: {len(result_df)}")
    print(f"保存到: {output_file}")
    
    return True

def copy_non_csv_files(source_dir: str, target_dir: str):
    """复制非CSV文件（如.txt文件）"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    for item in source_path.iterdir():
        if item.is_file() and not item.name.endswith('.csv'):
            target_file = target_path / item.name
            target_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_file)
            print(f"复制文件: {item} -> {target_file}")

def process_response_directory(input_dir: str, output_base: str):
    """处理单个response_*目录"""
    input_path = Path(input_dir)
    dir_name = input_path.name
    output_dir = Path(output_base) / dir_name
    
    print(f"\n处理目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 遍历所有子目录和文件
    for root, dirs, files in os.walk(input_dir):
        relative_path = Path(root).relative_to(input_path)
        target_root = output_dir / relative_path
        
        # 处理CSV文件
        for file in files:
            source_file = Path(root) / file
            target_file = target_root / file
            
            if file.endswith('.csv'):
                process_csv_file(str(source_file), str(target_file))
            else:
                # 复制非CSV文件
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_file)
                print(f"复制文件: {source_file} -> {target_file}")

def main():
    parser = argparse.ArgumentParser(description='将响应数据从5条采样到3条')
    parser.add_argument('--input_dir', default='results', help='输入目录路径')
    parser.add_argument('--output_dir', default='results_3', help='输出目录路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_random_seed(args.seed)
    
    # 确保输入目录存在
    if not Path(args.input_dir).exists():
        print(f"输入目录不存在: {args.input_dir}")
        return
    
    # 找到所有response_*目录
    response_dirs = find_response_directories(args.input_dir)
    
    if not response_dirs:
        print(f"在 {args.input_dir} 中没有找到response_*目录")
        return
    
    print(f"找到 {len(response_dirs)} 个response目录:")
    for dir_path in response_dirs:
        print(f"  - {dir_path}")
    
    # 处理每个response_*目录
    for response_dir in response_dirs:
        try:
            process_response_directory(response_dir, args.output_dir)
        except Exception as e:
            print(f"处理目录失败 {response_dir}: {e}")
            continue
    
    print(f"\n处理完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
