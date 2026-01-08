"""
SIUO 数据集适配器
"""

import json
from typing import Dict, Any, Iterator
from .base_adapter import BaseAdapter


class SIUOAdapter(BaseAdapter):
    """SIUO数据集适配器"""
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载SIUO原始数据"""
        # 加载siuo_gen.json文件
        json_file = 'siuo_gen.json'
        json_path = self.raw_data_path / json_file
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    yield item
        else:
            print(f"警告: 找不到文件 {json_path}")
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换SIUO样本
        
        原始格式:
        {
            "question_id": 1001,
            "image": "S-01.png",
            "question": "...",
            "category": "self-harm",
            "safety_warning": "...",
            "reference_answer": "..."
        }
        """
        # 获取图片路径
        image_filename = raw_sample.get('image', '')
        if image_filename:
            # 图片在images/子目录下
            abs_image_path = self.resolve_image_path(f'images/{image_filename}')
        else:
            abs_image_path = ""
        
        # 构建prompt
        prompt = raw_sample.get('question', '')
        
        # 构建meta信息
        meta = {
            'dataset': 'SIUO',
            'question_id': raw_sample.get('question_id', ''),
            'category': raw_sample.get('category', ''),
            'safety_warning': raw_sample.get('safety_warning', ''),
            'reference_answer': raw_sample.get('reference_answer', ''),
            'task_type': 'safety',
            'eval_metric': 'safety_score'
        }
        
        return {
            'prompt': prompt,
            'images': [abs_image_path] if abs_image_path else [],
            'meta': meta
        }

