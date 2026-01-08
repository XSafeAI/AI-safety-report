"""
基础适配器类
所有数据集适配器的基类
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator
from pathlib import Path


class BaseAdapter(ABC):
    """数据集适配器基类"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        初始化适配器
        
        Args:
            raw_data_path: 原始数据路径
            processed_data_path: 处理后数据保存路径
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # 确保输出目录存在
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """
        加载原始数据
        
        Returns:
            原始数据迭代器
        """
        pass
    
    @abstractmethod
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将单个样本转换为统一格式
        
        Args:
            raw_sample: 原始样本数据
            
        Returns:
            标准格式的样本: {
                "prompt": "用户看到的完整指令",
                "images": [img1, img2, ...],
                "meta": {... ground_truth labels ...}
            }
        """
        pass
    
    def process(self):
        """
        处理数据集并保存为jsonl格式
        """
        print(f"开始处理数据集: {self.raw_data_path}")
        
        count = 0
        with open(self.processed_data_path, 'w', encoding='utf-8') as f:
            for raw_sample in self.load_raw_data():
                try:
                    converted_sample = self.convert_sample(raw_sample)
                    if converted_sample:  # 跳过None结果
                        f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        if count % 100 == 0:
                            print(f"已处理 {count} 条样本...")
                except Exception as e:
                    print(f"处理样本时出错: {e}")
                    print(f"问题样本: {raw_sample}")
                    continue
        
        print(f"处理完成! 共处理 {count} 条样本")
        print(f"输出文件: {self.processed_data_path}")
    
    def resolve_image_path(self, relative_path: str) -> str:
        """
        解析图片相对路径为绝对路径
        
        Args:
            relative_path: 相对路径（相对于raw_data_path）
            
        Returns:
            绝对路径字符串
        """
        if not relative_path:
            return ""
        
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(relative_path):
            return relative_path
        
        # 拼接为绝对路径（相对于当前数据集的raw_data_path）
        abs_path = self.raw_data_path / relative_path
        return str(abs_path)

