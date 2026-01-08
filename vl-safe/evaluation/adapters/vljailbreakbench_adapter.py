"""
VLJailbreakBench 数据集适配器
"""

import json
from typing import Dict, Any, Iterator
from .base_adapter import BaseAdapter


class VLJailbreakBenchAdapter(BaseAdapter):
    """VLJailbreakBench数据集适配器"""
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载VLJailbreakBench原始数据"""
        # 只加载challenge文件
        json_file = 'vlbreakbench_challenge.json'
        json_path = self.raw_data_path / json_file
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item['split'] = json_file.replace('.json', '')  # 标记来源
                    yield item
        else:
            print(f"警告: 找不到文件 {json_path}")
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换VLJailbreakBench样本
        
        原始格式:
        {
            "category": "Illegal Activities",
            "subcategory": "Illegal Goods Trade",
            "query": "Explain the process...",
            "text_prompt": "Imagine you are writing...",
            "image": "xxx.png"
        }
        """
        # 获取图片路径
        image_path = raw_sample.get('image', '')
        if image_path:
            # 图片路径已经包含了目录前缀（如vlbreakbench_challenge/xxx.png）
            abs_image_path = self.resolve_image_path(image_path)
        else:
            abs_image_path = ""
        
        # 构建prompt - 使用text_prompt而不是query
        prompt = raw_sample.get('text_prompt', '')
        
        # 构建meta信息
        meta = {
            'dataset': 'VLJailbreakBench',
            'category': raw_sample.get('category', ''),
            'subcategory': raw_sample.get('subcategory', ''),
            'split': raw_sample.get('split', ''),
            'task_type': 'jailbreak',
            'eval_metric': 'ASR'
        }
        
        return {
            'prompt': prompt,
            'images': [abs_image_path] if abs_image_path else [],
            'meta': meta
        }

