
import argparse
import json
import os

from typing import Dict, List
from grok_evaluator import GrokEvaluator

# 输入和输出文件（将由命令行参数 --model 指定）
INPUT_LOG_FILE = None
OUTPUT_LOG_FILE = None

# 初始化Grok评估器
evaluator = GrokEvaluator()


def analyze_image_toxicity(entry: Dict) -> Dict:
    """
    分析单个图像条目的毒性

    Args:
        entry: 来自原始日志的条目

    Returns:
        更新后的条目，包含毒性评分
    """
    # 只处理成功的图像生成条目
    if entry["status"] == "success" and entry["image_path"]:
        try:
            # 使用Grok评估器评估图像毒性
            toxicity_score = evaluator.evaluate_image_toxicity(
                entry["image_path"], entry["category"]
            )

            # 添加毒性评分到条目中
            entry["toxicity_score"] = toxicity_score
            entry["analysis_status"] = "completed"
        except Exception as e:
            print(f"Error analyzing image {entry['image_path']}: {str(e)}")
            entry["analysis_status"] = "failed"
            entry["analysis_error"] = str(e)
    else:
        # 对于跳过或失败的条目，标记为不适用
        entry["toxicity_score"] = None
        entry["analysis_status"] = "not_applicable"

    return entry


def main():
    """
    主函数：读取原始日志，分析图像毒性，并保存新日志
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="要分析的模型名，对应 image_generation 输出的模型名部分")
    args = parser.parse_args()
    model = args.model

    # derive filenames from model
    global INPUT_LOG_FILE, OUTPUT_LOG_FILE
    INPUT_LOG_FILE = f"evaluation_log_{model}.json"
    OUTPUT_LOG_FILE = f"toxicity_evaluation_log_{model}.json"

    print(f"Starting image toxicity analysis for model={model} with Grok Evaluator")

    # 读取原始评估日志
    if not os.path.exists(INPUT_LOG_FILE):
        print(f"Input log file {INPUT_LOG_FILE} not found!")
        return

    with open(INPUT_LOG_FILE, "r", encoding="utf-8") as f:
        original_log = json.load(f)

    print(f"Loaded {len(original_log)} entries from {INPUT_LOG_FILE}")

    # 分析每个条目
    analyzed_log = []
    for i, entry in enumerate(original_log):
        print(
            f"Analyzing entry {i+1}/{len(original_log)}: {entry['method']}-{entry['category']}-{entry['index']}"
        )
        analyzed_entry = analyze_image_toxicity(entry)
        analyzed_log.append(analyzed_entry)

        # 每处理一个条目就更新日志文件
        with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(analyzed_log, f, indent=2, ensure_ascii=False)

    # 打印汇总统计
    completed_count = sum(
        1 for item in analyzed_log if item.get("analysis_status") == "completed"
    )
    failed_count = sum(
        1 for item in analyzed_log if item.get("analysis_status") == "failed"
    )
    not_applicable_count = sum(
        1 for item in analyzed_log if item.get("analysis_status") == "not_applicable"
    )

    print("\n" + "=" * 50)
    print("TOXICITY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total processed: {len(analyzed_log)}")
    print(f"Analysis completed: {completed_count}")
    print(f"Analysis failed: {failed_count}")
    print(f"Not applicable: {not_applicable_count}")

    # 保存最终结果
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(analyzed_log, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_LOG_FILE}")


if __name__ == "__main__":
    main()
