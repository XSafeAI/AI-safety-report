"""
JailbreakV-28K 数据集适配器
"""

import csv
import os
from typing import Dict, Any, Iterator
from .base_adapter import BaseAdapter


class JailbreakVAdapter(BaseAdapter):
    """JailbreakV-28K数据集适配器"""
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载JailbreakV-28K原始数据"""
        # 使用完整的CSV文件
        csv_path = self.raw_data_path / 'JailBreakV_28K' / 'JailBreakV_28K.csv'
        
        if not csv_path.exists():
            print(f"警告: 找不到文件 {csv_path}")
            return
        
        skipped_count = 0
        total_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                
                # 检查图片是否存在
                image_path = row.get('image_path', '')
                if image_path and image_path.strip():
                    # 构建完整的图片路径
                    full_image_path = self.raw_data_path / 'JailBreakV_28K' / image_path
                    
                    # 只有图片存在时才yield这条数据
                    if full_image_path.exists():
                        yield row
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
        
        print(f"JailbreakV-28K: 总共 {total_count} 条数据，跳过 {skipped_count} 条（图片不存在），保留 {total_count - skipped_count} 条")
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换JailbreakV样本
        
        原始格式:
        {
            "jailbreak_query": "... adversarial prompt ...",
            "redteam_query": "true harmful request",
            "policy": "violence",
            "format": "template",
            "image_path": "images/xxx.jpg"
        }
        """
        # 获取图片路径
        image_path = raw_sample.get('image_path', '')
        if image_path and image_path.strip():
            # 图片在JailBreakV_28K目录下
            abs_image_path = self.resolve_image_path(f"JailBreakV_28K/{image_path}")
        else:
            abs_image_path = ""
        
        # 使用jailbreak_query作为prompt
        prompt = raw_sample.get('jailbreak_query', '')
        
        # 构建meta信息
        meta = {
            'dataset': 'JailbreakV-28K',
            'redteam_query': raw_sample.get('redteam_query', ''),
            'policy': raw_sample.get('policy', ''),
            'format': raw_sample.get('format', ''),
            'from': raw_sample.get('from', ''),
            'task_type': 'jailbreak',
            'eval_metric': 'ASR'
        }
        
        return {
            'prompt': prompt,
            'images': [abs_image_path] if abs_image_path else [],
            'meta': meta
        }

