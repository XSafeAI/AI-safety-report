"""
生成数据处理统计报告
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def generate_report(processed_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"):
    """
    生成处理后数据的统计报告
    
    Args:
        processed_root: 处理后数据根目录
    """
    processed_dir = Path(processed_root)
    
    if not processed_dir.exists():
        print(f"错误: 目录不存在 {processed_root}")
        return
    
    # 找到所有jsonl文件
    jsonl_files = sorted(processed_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"警告: 在 {processed_root} 中没有找到任何 .jsonl 文件")
        return
    
    print("="*100)
    print(" "*35 + "数据集处理统计报告")
    print("="*100)
    print()
    
    total_samples = 0
    dataset_stats = []
    
    for jsonl_file in jsonl_files:
        stats = analyze_single_dataset(jsonl_file)
        dataset_stats.append(stats)
        total_samples += stats['total_samples']
    
    # 打印汇总表格
    print("\n" + "="*100)
    print("汇总统计")
    print("="*100)
    print()
    print(f"{'数据集':<25} {'样本数':>10} {'有图片':>10} {'无图片':>10} {'多图':>10} {'任务类型':<20}")
    print("-"*100)
    
    for stats in dataset_stats:
        print(f"{stats['dataset_name']:<25} "
              f"{stats['total_samples']:>10,} "
              f"{stats['with_images']:>10,} "
              f"{stats['without_images']:>10,} "
              f"{stats['multi_image']:>10,} "
              f"{stats['main_task_type']:<20}")
    
    print("-"*100)
    print(f"{'总计':<25} {total_samples:>10,}")
    print("="*100)
    print()
    
    # 详细统计
    print("\n" + "="*100)
    print("详细统计")
    print("="*100)
    
    for stats in dataset_stats:
        print(f"\n{'='*100}")
        print(f"数据集: {stats['dataset_name']}")
        print(f"{'='*100}")
        print(f"总样本数: {stats['total_samples']:,}")
        print(f"包含图片的样本: {stats['with_images']:,} ({stats['with_images']/stats['total_samples']*100:.1f}%)")
        print(f"不包含图片的样本: {stats['without_images']:,} ({stats['without_images']/stats['total_samples']*100:.1f}%)")
        print(f"多图样本: {stats['multi_image']:,} ({stats['multi_image']/stats['total_samples']*100:.1f}%)")
        
        if stats['categories']:
            print(f"\n类别分布 (前10):")
            for cat, count in list(stats['categories'].most_common(10)):
                print(f"  {cat:<40}: {count:>6,} ({count/stats['total_samples']*100:.1f}%)")
        
        if stats['task_types']:
            print(f"\n任务类型分布:")
            for task, count in stats['task_types'].most_common():
                print(f"  {task:<40}: {count:>6,} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n" + "="*100)
    print("报告生成完成!")
    print("="*100)


def analyze_single_dataset(jsonl_path: Path):
    """
    分析单个数据集
    
    Args:
        jsonl_path: jsonl文件路径
        
    Returns:
        统计信息字典
    """
    total = 0
    with_images = 0
    without_images = 0
    multi_image = 0
    categories = Counter()
    task_types = Counter()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                total += 1
                
                # 统计图片
                images = sample.get('images', [])
                if images:
                    with_images += 1
                    if len(images) > 1:
                        multi_image += 1
                else:
                    without_images += 1
                
                # 统计类别
                meta = sample.get('meta', {})
                if 'category' in meta:
                    categories[meta['category']] += 1
                
                # 统计任务类型
                if 'task_type' in meta:
                    task_types[meta['task_type']] += 1
                    
            except json.JSONDecodeError:
                continue
    
    # 获取主要任务类型
    main_task_type = task_types.most_common(1)[0][0] if task_types else 'unknown'
    
    return {
        'dataset_name': jsonl_path.stem,
        'total_samples': total,
        'with_images': with_images,
        'without_images': without_images,
        'multi_image': multi_image,
        'categories': categories,
        'task_types': task_types,
        'main_task_type': main_task_type
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成数据处理统计报告')
    parser.add_argument('--processed-root', type=str,
                       default='/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed',
                       help='处理后数据根目录')
    
    args = parser.parse_args()
    
    generate_report(args.processed_root)

