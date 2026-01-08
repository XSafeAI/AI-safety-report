"""
验证处理后数据集的图片路径
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def check_dataset_images(jsonl_path, num_samples=5):
    """
    检查数据集的图片路径
    
    Args:
        jsonl_path: JSONL文件路径
        num_samples: 检查的样本数量
    
    Returns:
        检查结果字典
    """
    dataset_name = Path(jsonl_path).stem
    results = {
        'dataset': dataset_name,
        'total_checked': 0,
        'exists': 0,
        'not_exists': 0,
        'empty_images': 0,
        'examples': []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                images = sample.get('images', [])
                
                if not images:
                    results['empty_images'] += 1
                    results['examples'].append({
                        'index': i,
                        'images': [],
                        'status': 'empty'
                    })
                    continue
                
                results['total_checked'] += 1
                
                # 检查每个图片路径
                all_exist = True
                image_status = []
                for img_path in images:
                    exists = os.path.exists(img_path)
                    image_status.append({
                        'path': img_path,
                        'exists': exists
                    })
                    if exists:
                        results['exists'] += 1
                    else:
                        results['not_exists'] += 1
                        all_exist = False
                
                results['examples'].append({
                    'index': i,
                    'images': image_status,
                    'all_exist': all_exist
                })
                
            except json.JSONDecodeError:
                continue
    
    return results


def main():
    """主函数"""
    processed_dir = Path('/data/data-pool/dingyifan/VL-Safe/workspace/data/processed')
    
    print("="*100)
    print(" "*35 + "图片路径验证报告")
    print("="*100)
    print()
    
    all_results = []
    
    # 检查所有jsonl文件
    for jsonl_file in sorted(processed_dir.glob('*.jsonl')):
        print(f"\n检查数据集: {jsonl_file.stem}")
        print("-"*100)
        
        results = check_dataset_images(jsonl_file, num_samples=3)
        all_results.append(results)
        
        print(f"检查样本数: {results['total_checked']}")
        print(f"空图片样本: {results['empty_images']}")
        print(f"图片存在: {results['exists']}")
        print(f"图片不存在: {results['not_exists']}")
        
        # 打印示例
        print(f"\n前3个样本:")
        for example in results['examples']:
            if example.get('status') == 'empty':
                print(f"  样本 {example['index']}: 无图片")
            else:
                print(f"  样本 {example['index']}:")
                for img_info in example['images']:
                    status = "✓" if img_info['exists'] else "✗"
                    print(f"    {status} {img_info['path']}")
    
    # 汇总统计
    print("\n" + "="*100)
    print("汇总统计")
    print("="*100)
    print(f"\n{'数据集':<25} {'检查样本':>10} {'存在':>10} {'不存在':>10} {'空图片':>10}")
    print("-"*100)
    
    for results in all_results:
        print(f"{results['dataset']:<25} "
              f"{results['total_checked']:>10} "
              f"{results['exists']:>10} "
              f"{results['not_exists']:>10} "
              f"{results['empty_images']:>10}")
    
    print("="*100)


if __name__ == '__main__':
    main()

