"""
MM-SafetyBench 数据集适配器
"""

import pandas as pd
from typing import Dict, Any, Iterator
from pathlib import Path
from PIL import Image
import io
from .base_adapter import BaseAdapter


class MMSafetyBenchAdapter(BaseAdapter):
    """MM-SafetyBench数据集适配器"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        super().__init__(raw_data_path, processed_data_path)
        # 创建图片保存目录
        self.image_save_dir = self.processed_data_path.parent / 'mm_safetybench_images'
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载MM-SafetyBench原始数据"""
        data_dir = self.raw_data_path / 'data'
        
        if not data_dir.exists():
            print(f"警告: 找不到目录 {data_dir}")
            return
        
        # 遍历所有类别目录
        for category_dir in data_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            
            # 查找该类别下的parquet文件
            parquet_files = list(category_dir.glob('*.parquet'))
            
            for parquet_file in parquet_files:
                try:
                    print(f"正在加载: {category_name}/{parquet_file.name}")
                    df = pd.read_parquet(parquet_file)
                    for _, row in df.iterrows():
                        row_dict = row.to_dict()
                        row_dict['category'] = category_name
                        row_dict['source_file'] = parquet_file.stem
                        yield row_dict
                except Exception as e:
                    print(f"解析文件出错 {parquet_file}: {e}")
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换MM-SafetyBench样本
        
        Parquet格式:
        {
            "id": "xxx",
            "question": "...",
            "image": bytes (JPEG/PNG binary data),
            "category": "EconomicHarm",
            "source_file": "SD"
        }
        """
        # 处理图片数据
        image = raw_sample.get('image', None)
        abs_image_path = ""
        
        if image is not None:
            try:
                # 如果是bytes，保存为文件
                if isinstance(image, bytes):
                    # 生成文件名
                    sample_id = raw_sample.get('id', 'unknown')
                    category = raw_sample.get('category', 'unknown')
                    source_file = raw_sample.get('source_file', 'unknown')
                    
                    # 创建子目录
                    category_dir = self.image_save_dir / category
                    category_dir.mkdir(exist_ok=True)
                    
                    # 保存图片
                    image_filename = f"{source_file}_{sample_id}.jpg"
                    image_path = category_dir / image_filename
                    
                    # 如果文件不存在才保存
                    if not image_path.exists():
                        # 使用PIL打开并保存
                        img = Image.open(io.BytesIO(image))
                        img.save(image_path)
                    
                    abs_image_path = str(image_path)
                    
                elif isinstance(image, str) and image:
                    # 如果是路径字符串
                    category = raw_sample.get('category', 'unknown')
                    abs_image_path = self.resolve_image_path(f"data/{category}/{image}")
            except Exception as e:
                print(f"处理图片失败: {e}")
                abs_image_path = ""
        
        # 获取问题/查询
        prompt = raw_sample.get('question', '')
        if not prompt:
            prompt = raw_sample.get('query', '')
        
        # 构建meta信息
        meta = {
            'dataset': 'MM-SafetyBench',
            'category': raw_sample.get('category', ''),
            'source_file': raw_sample.get('source_file', ''),
            'task_type': 'safety_detection',
            'eval_metric': 'ASR/Safety_Rate'
        }
        
        # 添加其他字段
        for key in ['id', 'subcategory', 'risk_level', 'answer', 'label']:
            if key in raw_sample:
                meta[key] = raw_sample[key]
        
        # 如果images为空，剔除该数据
        if not abs_image_path:
            return None
        
        return {
            'prompt': prompt,
            'images': [abs_image_path],
            'meta': meta
        }

