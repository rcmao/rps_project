import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np

def analyze_single_file(file_path):
    """分析单个JSONL结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 统计函数
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
    
    # 统计各类结果
    outcomes = [judge_winner(item) for item in data]
    counter = Counter(outcomes)
    
    total = sum(counter.values())
    win_fdiv = counter.get("f-div", 0)
    win_baseline = counter.get("baseline", 0)
    tie = counter.get("Tie", 0)
    unclear = counter.get("Unclear", 0)
    error = counter.get("Error", 0)
    
    # 计算百分比
    fdiv_rate = round(win_fdiv / total * 100, 2) if total > 0 else 0
    baseline_rate = round(win_baseline / total * 100, 2) if total > 0 else 0
    tie_rate = round(tie / total * 100, 2) if total > 0 else 0
    error_rate = round((unclear + error) / total * 100, 2) if total > 0 else 0
    
    return {
        "total": total,
        "f-div_wins": win_fdiv,
        "baseline_wins": win_baseline,
        "ties": tie,
        "unclear": unclear,
        "error": error,
        "f-div_rate": fdiv_rate,
        "baseline_rate": baseline_rate,
        "tie_rate": tie_rate,
        "error_rate": error_rate
    }

def analyze_all_results(results_dir="compare_result"):
    """分析所有结果文件"""
    results = {}
    
    # 定义向量到角度的映射
    vector_angles = {
        "v3": 10,
        "v4": 15, 
        "v5": 20,
        "v6": 25,
        "v7": 30,
        "v8": 35,
        "v9": 40,
        "v10": 45
    }
    
    # 分析每个文件
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.jsonl"):
            # 提取向量名称
            vector_name = None
            for v in vector_angles.keys():
                if filename.startswith(v):
                    vector_name = v
                    break
            
            if vector_name:
                file_path = os.path.join(results_dir, filename)
                result = analyze_single_file(file_path)
                result["vector"] = vector_name
                result["angle"] = vector_angles[vector_name]
                result["filename"] = filename
                results[vector_name] = result
    
    return results

def create_summary_table(results):
    """创建汇总表格"""
    df_data = []
    for vector, result in sorted(results.items(), key=lambda x: x[1]["angle"]):
        df_data.append({
            "向量": vector,
            "角度": f"{result['angle']}°",
            "总数": result["total"],
            "RPS胜利": result["f-div_wins"],
            "Baseline胜利": result["baseline_wins"],
            "平局": result["ties"],
            "RPS胜率(%)": result["f-div_rate"],
            "Baseline胜率(%)": result["baseline_rate"],
            "平局率(%)": result["tie_rate"],
            "错误率(%)": result["error_rate"]
        })
    
    df = pd.DataFrame(df_data)
    return df

def plot_results(results):
    """绘制结果图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    vectors = []
    angles = []
    fdiv_rates = []
    baseline_rates = []
    tie_rates = []
    
    for vector, result in sorted(results.items(), key=lambda x: x[1]["angle"]):
        vectors.append(vector)
        angles.append(result["angle"])
        fdiv_rates.append(result["f-div_rate"])
        baseline_rates.append(result["baseline_rate"])
        tie_rates.append(result["tie_rate"])
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 胜率对比柱状图
    x = np.arange(len(vectors))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, fdiv_rates, width, label='RPS', color='#2ca02c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, baseline_rates, width, label='Baseline', color='#1f77b4', alpha=0.8)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Vector Direction')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('RPS vs Baseline Win Rate Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(vectors)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 随角度变化的趋势线
    ax2.plot(angles, fdiv_rates, 'o-', label='RPS', color='#2ca02c', linewidth=2, markersize=8)
    ax2.plot(angles, baseline_rates, 's-', label='Baseline', color='#1f77b4', linewidth=2, markersize=8)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate Trend by Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 堆叠柱状图显示所有结果分布
    bottom1 = np.zeros(len(vectors))
    bottom2 = np.array(fdiv_rates)
    bottom3 = bottom2 + np.array(baseline_rates)
    
    ax3.bar(vectors, fdiv_rates, label='RPS Wins', color='#2ca02c', alpha=0.8)
    ax3.bar(vectors, baseline_rates, bottom=bottom2, label='Baseline Wins', color='#1f77b4', alpha=0.8)
    ax3.bar(vectors, tie_rates, bottom=bottom3, label='Ties', color='#ff7f0e', alpha=0.8)
    
    ax3.set_xlabel('Vector Direction')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Result Distribution (Stacked)')
    ax3.legend()
    
    # 4. RPS优势分析（RPS胜率 - Baseline胜率）
    advantage = np.array(fdiv_rates) - np.array(baseline_rates)
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in advantage]
    bars4 = ax4.bar(vectors, advantage, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, adv in zip(bars4, advantage):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{adv:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Vector Direction')
    ax4.set_ylabel('RPS Advantage (%)')
    ax4.set_title('RPS Relative Advantage (Positive = RPS Better)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_analysis(results):
    """生成详细分析报告"""
    print("="*80)
    print("RPS (Angle-Based Perturbation) vs Baseline Experiment Analysis")
    print("="*80)
    
    # 整体统计
    total_tests = sum(r["total"] for r in results.values())
    total_fdiv_wins = sum(r["f-div_wins"] for r in results.values())
    total_baseline_wins = sum(r["baseline_wins"] for r in results.values())
    total_ties = sum(r["ties"] for r in results.values())
    
    overall_fdiv_rate = round(total_fdiv_wins / total_tests * 100, 2)
    overall_baseline_rate = round(total_baseline_wins / total_tests * 100, 2)
    overall_tie_rate = round(total_ties / total_tests * 100, 2)
    
    print(f"\n【OVERALL STATISTICS】")
    print(f"Total Tests: {total_tests}")
    print(f"RPS Total Wins: {total_fdiv_wins} ({overall_fdiv_rate}%)")
    print(f"Baseline Total Wins: {total_baseline_wins} ({overall_baseline_rate}%)")
    print(f"Ties: {total_ties} ({overall_tie_rate}%)")
    
    # 分析最佳和最差表现
    sorted_results = sorted(results.items(), key=lambda x: x[1]["f-div_rate"] - x[1]["baseline_rate"], reverse=True)
    
    print(f"\n【BEST RPS PERFORMANCE】")
    best_vector, best_result = sorted_results[0]
    advantage = best_result["f-div_rate"] - best_result["baseline_rate"]
    print(f"{best_vector} ({best_result['angle']}°): RPS {best_result['f-div_rate']}% vs Baseline {best_result['baseline_rate']}% (Advantage: +{advantage:.1f}%)")
    
    print(f"\n【WORST RPS PERFORMANCE】")
    worst_vector, worst_result = sorted_results[-1]
    disadvantage = worst_result["baseline_rate"] - worst_result["f-div_rate"]
    print(f"{worst_vector} ({worst_result['angle']}°): RPS {worst_result['f-div_rate']}% vs Baseline {worst_result['baseline_rate']}% (Disadvantage: -{disadvantage:.1f}%)")
    
    # 角度区间分析
    print(f"\n【ANGLE RANGE ANALYSIS】")
    small_angles = [v for v, r in results.items() if r["angle"] <= 20]  # v3-v5
    medium_angles = [v for v, r in results.items() if 25 <= r["angle"] <= 35]  # v6-v8
    large_angles = [v for v, r in results.items() if r["angle"] >= 40]  # v9-v10
    
    for angle_range, vectors in [("Small Angles(10°-20°)", small_angles), 
                                ("Medium Angles(25°-35°)", medium_angles), 
                                ("Large Angles(40°-45°)", large_angles)]:
        if vectors:
            avg_fdiv = np.mean([results[v]["f-div_rate"] for v in vectors])
            avg_baseline = np.mean([results[v]["baseline_rate"] for v in vectors])
            print(f"{angle_range}: RPS avg {avg_fdiv:.1f}%, Baseline avg {avg_baseline:.1f}%")
    
    # RPS获胜的角度数量
    rps_wins = sum(1 for r in results.values() if r["f-div_rate"] > r["baseline_rate"])
    print(f"\n【SUMMARY】")
    print(f"RPS performs better in {rps_wins}/{len(results)} angles")
    if rps_wins > len(results) / 2:
        print("✅ RPS method overall outperforms Baseline")
    else:
        print("❌ RPS method overall underperforms Baseline")
    
    # 详细的每个角度分析
    print(f"\n【DETAILED BREAKDOWN】")
    for vector, result in sorted(results.items(), key=lambda x: x[1]["angle"]):
        advantage = result["f-div_rate"] - result["baseline_rate"]
        status = "✅" if advantage > 0 else "❌" if advantage < 0 else "⚖️"
        print(f"{status} {vector} ({result['angle']}°): RPS {result['f-div_rate']}% | Baseline {result['baseline_rate']}% | Tie {result['tie_rate']}% | Advantage: {advantage:+.1f}%")

def main():
    """主函数"""
    print("Starting experiment results analysis...")
    
    # 分析所有结果
    results = analyze_all_results()
    
    if not results:
        print("❌ No result files found, please check compare_result directory")
        return
    
    print(f"✅ Successfully loaded {len(results)} result files")
    
    # 创建汇总表格
    summary_df = create_summary_table(results)
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 保存表格到CSV
    summary_df.to_csv("experiment_results_summary.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ Summary table saved as experiment_results_summary.csv")
    
    # 生成详细分析
    generate_detailed_analysis(results)
    
    # 绘制图表
    print(f"\n📊 Generating visualization charts...")
    plot_results(results)
    print(f"✅ Charts saved as experiment_results_analysis.png")
    
    print(f"\n🎉 Analysis completed!")

if __name__ == "__main__":
    main() c