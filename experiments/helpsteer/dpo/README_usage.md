# HelpSteer DPO比较工具使用说明

改造后的 `compare_helpsteer_dpo_rps_vs_baseline.py` 现在支持交互式输入和限制比较的prompt数量。

## 使用方式

### 1. 交互式模式（推荐新用户）
```bash
python compare_helpsteer_dpo_rps_vs_baseline.py --interactive
```
程序会逐步引导你输入：
- Baseline目录路径
- RPS目录路径  
- 输出目录路径
- 模型名称（默认：gpt-4o-mini）
- 每个direction最大比较的prompt数量（可选，留空表示全部）

### 2. 快速启动模式
```bash
python compare_helpsteer_dpo_rps_vs_baseline.py
```
如果缺少必需参数，程序会自动提示输入。

### 3. 完整命令行模式
```bash
python compare_helpsteer_dpo_rps_vs_baseline.py \
    --baseline_dir /path/to/baseline/csvs \
    --rps_dir /path/to/rps/csvs \
    --output_dir /path/to/output \
    --model gpt-4o-mini \
    --max_prompts 50 \
    --seed 42
```

## 新功能

### 1. 交互式输入
- 支持路径验证
- 提供默认值选项
- 友好的用户界面
- 配置确认功能

### 2. Prompt数量限制
- `--max_prompts` 参数限制每个direction比较的prompt数量
- 用于快速测试或资源受限的情况
- 自动显示限制的prompt数量

### 3. 灵活的参数配置
- 所有参数都可以通过命令行或交互式输入
- 智能检测缺失参数并提示输入
- 支持混合模式（部分命令行 + 部分交互式）

## 示例运行

```bash
$ python compare_helpsteer_dpo_rps_vs_baseline.py --interactive

🚀 HelpSteer DPO: RPS vs Baseline Comparison Tool
============================================================
Enter baseline directory path: /data/baseline_results
Enter RPS directory path: /data/rps_results
Enter output directory path [./comparison_output]: ./my_comparison
Enter model name [gpt-4o-mini]: 
Enter max prompts per direction (leave empty for all): 100

📋 Configuration:
   Baseline dir: /data/baseline_results
   RPS dir: /data/rps_results
   Output dir: ./my_comparison
   Model: gpt-4o-mini
   Max prompts: 100
   Seed: 42

Proceed with this configuration? [Y/n]: y
```

## 环境要求

确保设置了以下环境变量：
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_API_BASE`: （可选）自定义API端点
