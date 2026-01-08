"""
Data loading utility class
Provides convenient dataset loading and access functions
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from PIL import Image


class DatasetLoader:
    """
    Dataset loader
    
    Used to load and access processed JSONL format datasets
    """
    
    def __init__(self, jsonl_path: str):
        """
        Initialize dataset loader
        
        Args:
            jsonl_path: Path to processed JSONL file
        """
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {jsonl_path}")
        
        # Cache dataset name and statistics
        self._dataset_name = self.jsonl_path.stem
        self._total_samples = None
    
    def __len__(self) -> int:
        """Return number of samples in dataset"""
        if self._total_samples is None:
            self._total_samples = sum(1 for _ in self.iter_samples())
        return self._total_samples
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through dataset samples
        
        Yields:
            Sample dictionary containing prompt, images, meta fields
        """
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line)
                    # Add line number as ID
                    sample['_id'] = line_num
                    yield sample
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing failed at line {line_num}: {e}")
                    continue
    
    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all samples into memory
        
        Returns:
            List of samples
        """
        return list(self.iter_samples())
    
    def get_sample(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get sample at specified index
        
        Args:
            index: Sample index (starting from 0)
            
        Returns:
            Sample dictionary, or None if index out of range
        """
        for i, sample in enumerate(self.iter_samples()):
            if i == index:
                return sample
        return None
    
    def iter_with_images(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through dataset samples and load images
        
        Yields:
            Sample dictionary, with images field changed from path list to PIL Image object list
        """
        for sample in self.iter_samples():
            # Load images
            loaded_images = []
            for img_path in sample.get('images', []):
                try:
                    img = Image.open(img_path)
                    loaded_images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue
            
            # Update sample
            sample['loaded_images'] = loaded_images
            yield sample
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Statistics dictionary
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
        Filter samples by category
        
        Args:
            category: Category name
            
        Yields:
            Matching samples
        """
        for sample in self.iter_samples():
            if sample.get('meta', {}).get('category') == category:
                yield sample
    
    def filter_by_task_type(self, task_type: str) -> Iterator[Dict[str, Any]]:
        """
        Filter samples by task type
        
        Args:
            task_type: Task type
            
        Yields:
            Matching samples
        """
        for sample in self.iter_samples():
            if sample.get('meta', {}).get('task_type') == task_type:
                yield sample


class MultiDatasetLoader:
    """
    Multi-dataset loader
    
    Used to load and manage multiple datasets simultaneously
    """
    
    def __init__(self, processed_root: str = "/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed"):
        """
        Initialize multi-dataset loader
        
        Args:
            processed_root: Processed data root directory
        """
        self.processed_root = Path(processed_root)
        self.datasets = {}
        
        # Automatically discover all JSONL files
        if self.processed_root.exists():
            for jsonl_file in self.processed_root.glob('*.jsonl'):
                dataset_name = jsonl_file.stem
                self.datasets[dataset_name] = DatasetLoader(str(jsonl_file))
    
    def __getitem__(self, dataset_name: str) -> DatasetLoader:
        """
        Get loader for specified dataset
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dataset loader
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset does not exist: {dataset_name}. Available datasets: {list(self.datasets.keys())}")
        return self.datasets[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """Return all available dataset names"""
        return list(self.datasets.keys())
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all datasets
        
        Returns:
            Dictionary mapping dataset names to statistics
        """
        return {
            name: loader.get_statistics()
            for name, loader in self.datasets.items()
        }
    
    def iter_all_samples(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through all samples in all datasets
        
        Yields:
            Sample dictionary
        """
        for dataset_name, loader in self.datasets.items():
            for sample in loader.iter_samples():
                yield sample


# Usage examples
if __name__ == '__main__':
    # Example 1: Load single dataset
    print("Example 1: Load single dataset")
    loader = DatasetLoader('/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed/vljailbreakbench.jsonl')
    
    print(f"Dataset: {loader._dataset_name}")
    print(f"Sample count: {len(loader)}")
    
    # Get first sample
    sample = loader.get_sample(0)
    if sample:
        print(f"\nFirst sample:")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        print(f"  Image count: {len(sample['images'])}")
        print(f"  Meta: {sample['meta']}")
    
    # Example 2: Statistics
    print("\n" + "="*80)
    print("Example 2: Statistics")
    stats = loader.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"With images: {stats['with_images']}")
    print(f"Task types: {stats['task_types']}")
    
    # Example 3: Load multiple datasets
    print("\n" + "="*80)
    print("Example 3: Load multiple datasets")
    multi_loader = MultiDatasetLoader()
    
    print(f"Available datasets: {multi_loader.list_datasets()}")
    
    # Access specific dataset
    if 'usb' in multi_loader.list_datasets():
        usb_loader = multi_loader['usb']
        print(f"\nUSB dataset sample count: {len(usb_loader)}")
    
    # Example 4: Filter samples
    print("\n" + "="*80)
    print("Example 4: Filter samples")
    count = 0
    for sample in loader.filter_by_task_type('jailbreak'):
        count += 1
        if count >= 3:  # Only show first 3
            break
    print(f"Filtered jailbreak samples: {count}")

