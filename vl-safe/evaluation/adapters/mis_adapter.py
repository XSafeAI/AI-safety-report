"""
MIS (Test) 数据集适配器
多图推理安全数据集
"""

import json
from typing import Dict, Any, Iterator
from .base_adapter import BaseAdapter


class MISAdapter(BaseAdapter):
    """MIS Test数据集适配器"""
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载MIS原始数据"""
        # 只加载hard文件
        json_file = 'mis_hard.json'
        json_path = self.raw_data_path / json_file
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                difficulty = 'hard'
                for item in data:
                    item['difficulty'] = difficulty
                    yield item
        else:
            print(f"警告: 找不到文件 {json_path}")
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换MIS样本
        
        原始格式:
        {
            "image1": "img1.jpg",
            "image2": "img2.jpg",
            "question": "...",
            "category": "safety",
            "sub_category": "implicit_violence"
        }
        """
        # 获取两张图片路径
        image_paths = []
        difficulty = raw_sample.get('difficulty', 'hard')
        
        # 图片可能在不同字段中
        for key in ['image_path1', 'image_path2', 'image1', 'image2', 'image_1', 'image_2', 'img1', 'img2']:
            image_path = raw_sample.get(key, '')
            if image_path:
                # JSON中的路径已经包含了hard_image/X/，只需加zip_files/前缀
                abs_path = self.resolve_image_path(f"zip_files/{image_path}")
                image_paths.append(abs_path)
        
        # 如果没有找到image_path1/image_path2，尝试images字段
        if not image_paths and 'images' in raw_sample:
            images = raw_sample['images']
            if isinstance(images, list):
                for img in images:
                    abs_path = self.resolve_image_path(f"zip_files/{img}")
                    image_paths.append(abs_path)
        
        # 构建prompt - 多图推理格式
        question = raw_sample.get('question', '')
        prompt = f"这是两张图片，请根据这两张图片回答问题：{question}"
        
        # 构建meta信息
        meta = {
            'dataset': 'MIS',
            'difficulty': difficulty,
            'category': raw_sample.get('category', ''),
            'sub_category': raw_sample.get('sub_category', ''),
            'question': question,
            'task_type': 'multi_image_reasoning',
            'eval_metric': 'Accuracy/ASR'
        }
        
        # 如果有答案字段，也保存
        if 'answer' in raw_sample:
            meta['answer'] = raw_sample['answer']
        
        return {
            'prompt': prompt,
            'images': image_paths,
            'meta': meta
        }

