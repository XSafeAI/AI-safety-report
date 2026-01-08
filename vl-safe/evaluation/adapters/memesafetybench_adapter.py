"""
MemeSafetyBench 数据集适配器
"""

import pandas as pd
from typing import Dict, Any, Iterator
from pathlib import Path
from PIL import Image
import io
from .base_adapter import BaseAdapter


class MemeSafetyBenchAdapter(BaseAdapter):
    """MemeSafetyBench数据集适配器"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str, use_mini: bool = True):
        """
        初始化适配器
        
        Args:
            raw_data_path: 原始数据路径
            processed_data_path: 处理后数据保存路径
            use_mini: 是否使用mini数据集（390条），否则使用full数据集（50K+条），默认True
        """
        super().__init__(raw_data_path, processed_data_path)
        self.use_mini = use_mini
        # 创建图片保存目录
        self.image_save_dir = self.processed_data_path.parent / 'memesafetybench_images'
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """加载MemeSafetyBench原始数据"""
        # 选择mini或full目录
        data_dir = self.raw_data_path / ('mini' if self.use_mini else 'full')
        
        if not data_dir.exists():
            print(f"警告: 找不到目录 {data_dir}")
            return
        
        # 读取所有parquet文件
        parquet_files = sorted(data_dir.glob('test-*.parquet'))
        
        for parquet_file in parquet_files:
            print(f"正在加载: {parquet_file.name}")
            df = pd.read_parquet(parquet_file)
            for _, row in df.iterrows():
                yield row.to_dict()
    
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换MemeSafetyBench样本
        
        Parquet中的格式:
        - meme_image: {'bytes': ..., 'path': ...}
        - image_path: 字符串路径
        - instruction: 指令文本
        - sentiment: 情感（positive/negative）
        - category: 类别
        """
        # 处理图片 - 从meme_image字段提取
        meme_image = raw_sample.get('meme_image', {})
        image_bytes = None
        image_path = ""
        
        # 尝试从meme_image字典中获取bytes
        if isinstance(meme_image, dict):
            image_bytes = meme_image.get('bytes')
            original_path = meme_image.get('path', '')
        
        # 如果有图片字节数据，保存为文件
        if image_bytes:
            try:
                # 生成唯一的文件名
                row_id = raw_sample.get('image_path', '').replace('/', '_').replace(' ', '_')
                if not row_id:
                    row_id = f"image_{hash(str(raw_sample))}"
                
                # 检测图片格式
                img = Image.open(io.BytesIO(image_bytes))
                ext = img.format.lower() if img.format else 'jpg'
                
                # 保存图片
                image_filename = f"{row_id}.{ext}"
                image_save_path = self.image_save_dir / image_filename
                
                # 如果文件不存在才保存
                if not image_save_path.exists():
                    img.save(image_save_path)
                
                image_path = str(image_save_path)
            except Exception as e:
                print(f"保存图片失败: {e}")
                image_path = ""
        
        # 获取指令
        instruction = raw_sample.get('instruction', '')
        if not instruction:
            instruction = raw_sample.get('question', '')
        
        prompt = instruction
        
        # 构建meta信息
        meta = {
            'dataset': 'MemeSafetyBench',
            'sentiment': raw_sample.get('sentiment', ''),
            'category': raw_sample.get('category', ''),
            'task': raw_sample.get('task', ''),
            'task_type': 'meme_safety',
            'eval_metric': 'RR/HR/CR'
        }
        
        # 添加其他可能的字段
        for key in ['label', 'harm_category', 'is_harmful']:
            if key in raw_sample:
                meta[key] = raw_sample[key]
        
        return {
            'prompt': prompt,
            'images': [image_path] if image_path else [],
            'meta': meta
        }

