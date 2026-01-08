"""
LLM模型评测脚本
使用统一的llm库对数据集进行推理评测
"""

import os
import json
import threading
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入统一的llm库
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import completion


class LLMEvaluator:
    """LLM评测器（支持多种模型）"""
    
    def __init__(self, model_name: str, concurrency: int = 5, 
                 reasoning_effort: str = "low", max_tokens: int = 256):
        """
        初始化评测器
        
        Args:
            model_name: 模型名称（如：gemini-3-pro-preview, gpt-5-mini, deepseek-reasoner等）
            concurrency: 并发数（1表示串行执行）
            reasoning_effort: 推理强度（"low", "medium", "high"，默认："low"）
            max_tokens: 最大生成token数（默认：256）
        """
        self.model_name = model_name
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        
        print(f"✓ 初始化LLM评测器")
        print(f"  模型: {model_name}")
        print(f"  并发数: {concurrency}")
        print(f"  推理强度: {reasoning_effort}")
        print(f"  最大tokens: {max_tokens}")
    
    def _build_messages(self, prompt: str, images: List[str]) -> List[Dict[str, Any]]:
        """
        构建OpenAI风格的messages格式
        
        Args:
            prompt: 文本提示
            images: 图片路径列表
            
        Returns:
            messages列表
        """
        # 如果没有图片，直接返回文本消息
        if not images:
            return [{"role": "user", "content": prompt}]
        
        # 构建多模态内容
        content = []
        
        # 添加图片
        for img_path in images:
            if os.path.exists(img_path):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_path  # 本地路径，会自动处理
                    }
                })
        
        # 添加文本
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    def generate(self, prompt: str, images: List[str], max_retries: int = 3) -> Dict[str, Any]:
        """
        生成响应（使用统一llm库，线程安全）
        
        Args:
            prompt: 文本提示
            images: 图片路径列表
            max_retries: 最大重试次数（默认3次）
            
        Returns:
            包含响应和元数据的字典
        """
        # 构建messages
        messages = self._build_messages(prompt, images)
        
        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
                max_tokens=self.max_tokens,
                retry_times=max_retries - 1,  # completion已有重试机制
                retry_delay=3.0
            )
            
            # # 处理返回值（可能是字符串或字典）
            # if isinstance(response, dict):
            #     response_text = response.get('content', '')
            # else:
            #     response_text = response
            
            return {
                'success': True,
                'response': response,
                'error': None,
                'attempts': 1
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': None,
                'error': str(e),
                'attempts': max_retries
            }
    
    def evaluate_batch_threaded(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        多线程批量评测
        
        Args:
            samples: 样本列表
            
        Returns:
            评测结果列表
        """
        # 创建进度条
        pbar = tqdm(total=len(samples), desc="评测进度", unit="样本")
        pbar_lock = threading.Lock()
        
        # 存储结果（按照原始顺序）
        results = [None] * len(samples)
        results_lock = threading.Lock()
        
        def process_sample(sample, idx):
            """处理单个样本"""
            prompt = sample.get('prompt', '')
            images = sample.get('images', [])
            meta = sample.get('meta', {})
                
                # 调用模型（带重试）
            result = self.generate(prompt, images, max_retries=3)
                
            # 更新进度条（线程安全）
            with pbar_lock:
                pbar.update(1)

                # 返回结果
                if not result.get('success', False):
                    result_data = {
                            'prompt': prompt,
                            'images': images,
                            'error': result.get('error'),
                            'meta': meta
                        }
                else:
                    result_data = {
                            'prompt': prompt,
                            'images': images,
                            'response': result.get('response'),
                            'meta': meta
                        }
        
            # 保存结果（线程安全）
            with results_lock:
                results[idx] = result_data
            
            return result_data
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            # 提交所有任务
            futures = [executor.submit(process_sample, sample, i) 
                      for i, sample in enumerate(samples)]
        
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                except Exception as e:
                    print(f"任务执行出错: {e}")
        
        # 关闭进度条
        pbar.close()
        
        return results
    
    def evaluate_batch_sequential(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        串行批量评测（不使用多线程）
        
        Args:
            samples: 样本列表
            
        Returns:
            评测结果列表
        """
        results = []
        
        # 使用tqdm显示进度
        for sample in tqdm(samples, desc="评测进度", unit="样本"):
            prompt = sample.get('prompt', '')
            images = sample.get('images', [])
            meta = sample.get('meta', {})
            
            # 调用模型（带重试）
            result = self.generate(prompt, images, max_retries=3)

            if not result.get('success', False):
                results.append({
                    'prompt': prompt,
                    'images': images,
                    'error': result.get('error'),
                    'meta': meta
                })
            else:
                results.append({
                    'prompt': prompt,
                    'images': images,
                    'response': result.get('response'),
                    'meta': meta
                })
        
        return results
    
    def evaluate_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        评测数据集（多线程）
        
        Args:
            dataset_path: 数据集JSONL文件路径
            max_samples: 最大样本数（用于调试）
            
        Returns:
            评测结果列表
        """
        print(f"\n开始评测数据集: {Path(dataset_path).stem}")
        
        # 加载数据集
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                sample = json.loads(line)
                samples.append(sample)
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        print(f"加载了 {len(samples)} 个样本")
        
        # 评测
        if self.concurrency == 1:
            print("使用串行执行模式...")
            results = self.evaluate_batch_sequential(samples)
        else:
            print(f"使用多线程执行模式（线程数: {self.concurrency}）...")
            results = self.evaluate_batch_threaded(samples)
        
        # 统计
        success_count = sum(1 for r in results if "response" in r)
        fail_count = len(results) - success_count
        
        print(f"\n评测完成!")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        
        return results
    
    
    @staticmethod
    def save_results(results: List[Dict[str, Any]], output_path: str):
        """
        保存评测结果
        
        Args:
            results: 评测结果列表
            output_path: 输出文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM模型评测脚本（支持多种模型）')
    parser.add_argument('--dataset', type=str, 
                       default='vljailbreakbench',
                       help='数据集名称（例如: vljailbreakbench, usb, mis_test等）')
    parser.add_argument('--max-samples', type=int, default=2,
                       help='最大评测样本数（用于调试，默认评测全部）')
    parser.add_argument('--concurrency', type=int, default=2,
                       help='并发数（设为1则串行执行，默认2）')
    parser.add_argument('--model', type=str, default='gemini-3-pro-preview',
                       help='模型名称（支持: gemini-3-pro-preview, gemini-2.5-pro, gpt-5-mini, deepseek-reasoner等）')
    parser.add_argument('--reasoning-effort', type=str, default='low',
                       choices=['low', 'medium', 'high'],
                       help='推理强度（low/medium/high，默认: low）')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='最大生成token数（默认: 256）')
    parser.add_argument('--processed-root', type=str,
                       default='/data/data-pool/dingyifan/VL-Safe/workspace/data/processed',
                       help='处理后数据根目录')
    parser.add_argument('--output-root', type=str,
                       default='/data/data-pool/dingyifan/VL-Safe/workspace/results',
                       help='结果输出根目录')
    
    args = parser.parse_args()
    
    # 加载环境变量（根据模型自动选择）
    env_path = Path('/data/data-pool/dingyifan/VL-Safe/.env')
    if env_path.exists():
        load_dotenv(env_path)
    
    # 构建数据集路径
    dataset_path = Path(args.processed_root) / f"{args.dataset}.jsonl"
    
    if not dataset_path.exists():
        print(f"错误: 找不到数据集文件: {dataset_path}")
        print(f"\n可用的数据集:")
        for jsonl_file in sorted(Path(args.processed_root).glob('*.jsonl')):
            print(f"  - {jsonl_file.stem}")
        return
    
    # 创建评测器
    evaluator = LLMEvaluator(
        model_name=args.model,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens
    )
    
    # 执行评测
    start_time = time.time()
    results = evaluator.evaluate_dataset(
        str(dataset_path),
        max_samples=args.max_samples
    )
    end_time = time.time()
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{args.dataset}_{timestamp}.jsonl"
    output_path = Path(args.output_root) / output_filename
    
    evaluator.save_results(results, str(output_path))
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("评测统计")
    print(f"{'='*80}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {len(results)}")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print(f"平均每样本: {(end_time - start_time) / len(results):.2f} 秒")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

