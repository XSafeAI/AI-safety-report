"""
Base adapter class
Base class for all dataset adapters
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterator
from pathlib import Path


class BaseAdapter(ABC):
    """Base class for dataset adapters"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize adapter
        
        Args:
            raw_data_path: Raw data path
            processed_data_path: Path to save processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Ensure output directory exists
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_raw_data(self) -> Iterator[Dict[str, Any]]:
        """
        Load raw data
        
        Returns:
            Iterator of raw data
        """
        pass
    
    @abstractmethod
    def convert_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single sample to unified format
        
        Args:
            raw_sample: Raw sample data
            
        Returns:
            Sample in standard format: {
                "prompt": "Complete instruction seen by user",
                "images": [img1, img2, ...],
                "meta": {... ground_truth labels ...}
            }
        """
        pass
    
    def process(self):
        """
        Process dataset and save in jsonl format
        """
        print(f"Starting to process dataset: {self.raw_data_path}")
        
        count = 0
        with open(self.processed_data_path, 'w', encoding='utf-8') as f:
            for raw_sample in self.load_raw_data():
                try:
                    converted_sample = self.convert_sample(raw_sample)
                    if converted_sample:  # Skip None results
                        f.write(json.dumps(converted_sample, ensure_ascii=False) + '\n')
                        count += 1
                        if count % 100 == 0:
                            print(f"Processed {count} samples...")
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    print(f"Problematic sample: {raw_sample}")
                    continue
        
        print(f"Processing complete! Total {count} samples processed")
        print(f"Output file: {self.processed_data_path}")
    
    def resolve_image_path(self, relative_path: str) -> str:
        """
        Resolve image relative path to absolute path
        
        Args:
            relative_path: Relative path (relative to raw_data_path)
            
        Returns:
            Absolute path string
        """
        if not relative_path:
            return ""
        
        # If already absolute path, return directly
        if os.path.isabs(relative_path):
            return relative_path
        
        # Concatenate to absolute path (relative to current dataset's raw_data_path)
        abs_path = self.raw_data_path / relative_path
        return str(abs_path)

