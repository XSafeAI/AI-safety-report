"""
数据集处理主脚本
将所有原始数据集转换为统一格式
"""

import argparse
from pathlib import Path
from adapters import (
    JailbreakVAdapter,
    MISAdapter,
    VLJailbreakBenchAdapter,
    USBAdapter,
    MemeSafetyBenchAdapter,
    MMSafetyBenchAdapter,
    SIUOAdapter
)


def process_all_datasets(
    raw_data_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/raw",
    processed_data_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"
):
    """
    处理所有数据集
    
    Args:
        raw_data_root: 原始数据根目录
        processed_data_root: 处理后数据根目录
    """
    raw_root = Path(raw_data_root)
    processed_root = Path(processed_data_root)
    
    # 确保输出目录存在
    processed_root.mkdir(parents=True, exist_ok=True)
    
    # 定义所有数据集及其适配器
    datasets = [
        {
            'name': 'VLJailbreakBench',
            'raw_path': raw_root / 'VLJailbreakBench',
            'processed_path': processed_root / 'vljailbreakbench.jsonl',
            'adapter_class': VLJailbreakBenchAdapter
        },
        {
            'name': 'JailbreakV-28K',
            'raw_path': raw_root / 'JailbreakV-28k',
            'processed_path': processed_root / 'jailbreakv_28k.jsonl',
            'adapter_class': JailbreakVAdapter
        },
        {
            'name': 'MIS',
            'raw_path': raw_root / 'MIS_Test',
            'processed_path': processed_root / 'mis_test.jsonl',
            'adapter_class': MISAdapter
        },
        {
            'name': 'USB',
            'raw_path': raw_root / 'USB',
            'processed_path': processed_root / 'usb.jsonl',
            'adapter_class': USBAdapter
        },
        {
            'name': 'MemeSafetyBench',
            'raw_path': raw_root / 'MemeSafetyBench',
            'processed_path': processed_root / 'memesafetybench.jsonl',
            'adapter_class': MemeSafetyBenchAdapter,
            'kwargs': {'use_mini': True}  # 使用mini数据集（390条）
        },
        {
            'name': 'MM-SafetyBench',
            'raw_path': raw_root / 'MM-SafetyBench',
            'processed_path': processed_root / 'mm_safetybench.jsonl',
            'adapter_class': MMSafetyBenchAdapter
        },
        {
            'name': 'SIUO',
            'raw_path': raw_root / 'SIUO',
            'processed_path': processed_root / 'siuo.jsonl',
            'adapter_class': SIUOAdapter
        }
    ]
    
    # 处理每个数据集
    for dataset_info in datasets:
        print("\n" + "="*80)
        print(f"开始处理数据集: {dataset_info['name']}")
        print("="*80)
        
        # 检查原始数据是否存在
        if not dataset_info['raw_path'].exists():
            print(f"警告: 找不到数据集目录 {dataset_info['raw_path']}")
            print("跳过该数据集...")
            continue
        
        try:
            # 创建适配器
            kwargs = dataset_info.get('kwargs', {})
            adapter = dataset_info['adapter_class'](
                str(dataset_info['raw_path']),
                str(dataset_info['processed_path']),
                **kwargs
            )
            
            # 处理数据
            adapter.process()
            
            print(f"✓ {dataset_info['name']} 处理完成!")
            
        except Exception as e:
            print(f"✗ 处理 {dataset_info['name']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("所有数据集处理完成!")
    print(f"处理后的数据保存在: {processed_root}")
    print("="*80)


def process_single_dataset(dataset_name: str, 
                          raw_data_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/raw",
                          processed_data_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"):
    """
    处理单个数据集
    
    Args:
        dataset_name: 数据集名称
        raw_data_root: 原始数据根目录
        processed_data_root: 处理后数据根目录
    """
    raw_root = Path(raw_data_root)
    processed_root = Path(processed_data_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    
    # 数据集映射
    dataset_map = {
        'vljailbreakbench': {
            'raw_path': raw_root / 'VLJailbreakBench',
            'processed_path': processed_root / 'vljailbreakbench.jsonl',
            'adapter_class': VLJailbreakBenchAdapter
        },
        'jailbreakv': {
            'raw_path': raw_root / 'JailbreakV-28k',
            'processed_path': processed_root / 'jailbreakv_28k.jsonl',
            'adapter_class': JailbreakVAdapter
        },
        'mis': {
            'raw_path': raw_root / 'MIS_Test',
            'processed_path': processed_root / 'mis_test.jsonl',
            'adapter_class': MISAdapter
        },
        'usb': {
            'raw_path': raw_root / 'USB',
            'processed_path': processed_root / 'usb.jsonl',
            'adapter_class': USBAdapter
        },
        'memesafetybench': {
            'raw_path': raw_root / 'MemeSafetyBench',
            'processed_path': processed_root / 'memesafetybench.jsonl',
            'adapter_class': MemeSafetyBenchAdapter,
            'kwargs': {'use_mini': True}  # 使用mini数据集（390条）
        },
        'mmsafetybench': {
            'raw_path': raw_root / 'MM-SafetyBench',
            'processed_path': processed_root / 'mm_safetybench.jsonl',
            'adapter_class': MMSafetyBenchAdapter
        },
        'siuo': {
            'raw_path': raw_root / 'SIUO',
            'processed_path': processed_root / 'siuo.jsonl',
            'adapter_class': SIUOAdapter
        }
    }
    
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in dataset_map:
        print(f"错误: 未知的数据集名称 '{dataset_name}'")
        print(f"支持的数据集: {', '.join(dataset_map.keys())}")
        return
    
    dataset_info = dataset_map[dataset_name_lower]
    
    print(f"开始处理数据集: {dataset_name}")
    
    if not dataset_info['raw_path'].exists():
        print(f"错误: 找不到数据集目录 {dataset_info['raw_path']}")
        return
    
    try:
        kwargs = dataset_info.get('kwargs', {})
        adapter = dataset_info['adapter_class'](
            str(dataset_info['raw_path']),
            str(dataset_info['processed_path']),
            **kwargs
        )
        adapter.process()
        print(f"✓ {dataset_name} 处理完成!")
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集适配器 - 将原始数据转换为统一格式')
    parser.add_argument('--dataset', type=str, default=None,
                       help='指定要处理的数据集名称（不指定则处理所有数据集）')
    parser.add_argument('--raw-root', type=str, 
                       default='/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/raw',
                       help='原始数据根目录')
    parser.add_argument('--processed-root', type=str,
                       default='/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed',
                       help='处理后数据根目录')
    
    args = parser.parse_args()
    
    if args.dataset:
        # 处理单个数据集
        process_single_dataset(args.dataset, args.raw_root, args.processed_root)
    else:
        # 处理所有数据集
        process_all_datasets(args.raw_root, args.processed_root)


if __name__ == '__main__':
    main()

