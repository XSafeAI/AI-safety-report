"""
数据加载工具类
提供便捷的数据集加载和访问功能
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from PIL import Image


class DatasetLoader:
    """
    数据集加载器
    
    用于加载和访问处理后的JSONL格式数据集
    """
    
    def __init__(self, jsonl_path: str):
        """
        初始化数据集加载器
        
        Args:
            jsonl_path: 处理后的JSONL文件路径
        """
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {jsonl_path}")
        
        # 缓存数据集名称和统计信息
        self._dataset_name = self.jsonl_path.stem
        self._total_samples = None
    
    def __len__(self) -> int:
        """返回数据集样本数量"""
        if self._total_samples is None:
            self._total_samples = sum(1 for _ in self.iter_samples())
        return self._total_samples
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """
        迭代访问数据集样本
        
        Yields:
            样本字典，包含prompt, images, meta字段
        """
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    # 添加行号作为ID
                    sample['_id'] = line_num
                    yield sample
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
    
    def load_all(self) -> List[Dict[str, Any]]:
        """
        加载所有样本到内存
        
        Returns:
            样本列表
        """
        return list(self.iter_samples())
    
    def get_sample(self, index: int) -> Optional[Dict[str, Any]]:
        """
        获取指定索引的样本
        
        Args:
            index: 样本索引（从0开始）
            
        Returns:
            样本字典，如果索引超出范围则返回None
        """
        for i, sample in enumerate(self.iter_samples()):
            if i == index:
                return sample
        return None
    
    def iter_with_images(self) -> Iterator[Dict[str, Any]]:
        """
        迭代访问数据集样本，并加载图片
        
        Yields:
            样本字典，images字段从路径列表变为PIL Image对象列表
        """
        for sample in self.iter_samples():
            # 加载图片
            loaded_images = []
            for img_path in sample.get('images', []):
                try:
                    img = Image.open(img_path)
                    loaded_images.append(img)
                except Exception as e:
                    print(f"警告: 图片加载失败 {img_path}: {e}")
                    continue
            
            # 更新样本
            sample['loaded_images'] = loaded_images
            yield sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        from collections import Counter
        
        total = 0
        with_images = 0
        without_images = 0
        multi_image = 0
        categories = Counter()
        task_types = Counter()
        
        for sample in self.iter_samples():
            total += 1
            
            images = sample.get('images', [])
            if images:
                with_images += 1
                if len(images) > 1:
                    multi_image += 1
            else:
                without_images += 1
            
            meta = sample.get('meta', {})
            if 'category' in meta:
                categories[meta['category']] += 1
            if 'task_type' in meta:
                task_types[meta['task_type']] += 1
        
        return {
            'dataset_name': self._dataset_name,
            'total_samples': total,
            'with_images': with_images,
            'without_images': without_images,
            'multi_image_samples': multi_image,
            'categories': dict(categories.most_common(10)),
            'task_types': dict(task_types)
        }
    
    def filter_by_category(self, category: str) -> Iterator[Dict[str, Any]]:
        """
        按类别筛选样本
        
        Args:
            category: 类别名称
            
        Yields:
            符合条件的样本
        """
        for sample in self.iter_samples():
            if sample.get('meta', {}).get('category') == category:
                yield sample
    
    def filter_by_task_type(self, task_type: str) -> Iterator[Dict[str, Any]]:
        """
        按任务类型筛选样本
        
        Args:
            task_type: 任务类型
            
        Yields:
            符合条件的样本
        """
        for sample in self.iter_samples():
            if sample.get('meta', {}).get('task_type') == task_type:
                yield sample


class MultiDatasetLoader:
    """
    多数据集加载器
    
    用于同时加载和管理多个数据集
    """
    
    def __init__(self, processed_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"):
        """
        初始化多数据集加载器
        
        Args:
            processed_root: 处理后数据根目录
        """
        self.processed_root = Path(processed_root)
        self.datasets = {}
        
        # 自动发现所有JSONL文件
        if self.processed_root.exists():
            for jsonl_file in self.processed_root.glob('*.jsonl'):
                dataset_name = jsonl_file.stem
                self.datasets[dataset_name] = DatasetLoader(str(jsonl_file))
    
    def __getitem__(self, dataset_name: str) -> DatasetLoader:
        """
        获取指定数据集的加载器
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集加载器
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"数据集不存在: {dataset_name}. 可用数据集: {list(self.datasets.keys())}")
        return self.datasets[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """返回所有可用数据集名称"""
        return list(self.datasets.keys())
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有数据集的统计信息
        
        Returns:
            数据集名称到统计信息的字典
        """
        return {
            name: loader.get_statistics()
            for name, loader in self.datasets.items()
        }
    
    def iter_all_samples(self) -> Iterator[Dict[str, Any]]:
        """
        迭代所有数据集的所有样本
        
        Yields:
            样本字典
        """
        for dataset_name, loader in self.datasets.items():
            for sample in loader.iter_samples():
                yield sample


# 使用示例
if __name__ == '__main__':
    # 示例1: 加载单个数据集
    print("示例1: 加载单个数据集")
    loader = DatasetLoader('/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed/vljailbreakbench.jsonl')
    
    print(f"数据集: {loader._dataset_name}")
    print(f"样本数: {len(loader)}")
    
    # 获取第一个样本
    sample = loader.get_sample(0)
    if sample:
        print(f"\n第一个样本:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  图片数量: {len(sample['images'])}")
        print(f"  Meta: {sample['meta']}")
    
    # 示例2: 统计信息
    print("\n" + "="*80)
    print("示例2: 统计信息")
    stats = loader.get_statistics()
    print(f"总样本数: {stats['total_samples']}")
    print(f"包含图片: {stats['with_images']}")
    print(f"任务类型: {stats['task_types']}")
    
    # 示例3: 加载多个数据集
    print("\n" + "="*80)
    print("示例3: 加载多个数据集")
    multi_loader = MultiDatasetLoader()
    
    print(f"可用数据集: {multi_loader.list_datasets()}")
    
    # 访问特定数据集
    if 'usb' in multi_loader.list_datasets():
        usb_loader = multi_loader['usb']
        print(f"\nUSB数据集样本数: {len(usb_loader)}")
    
    # 示例4: 筛选样本
    print("\n" + "="*80)
    print("示例4: 筛选样本")
    count = 0
    for sample in loader.filter_by_task_type('jailbreak'):
        count += 1
        if count >= 3:  # 只显示前3个
            break
    print(f"筛选出的jailbreak样本: {count}个")

