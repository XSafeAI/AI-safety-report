"""
LLM模型评测脚本
使用统一的llm库对数据集进行推理评测
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import time
from tqdm import tqdm

# 导入统一的llm库
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import completion, acompletion


class LLMEvaluator:
    """LLM评测器（支持多种模型）"""
    
    def __init__(self, model_name: str, concurrency: int = 5, 
                 reasoning_effort: str = "low", max_tokens: int = 256,
                 retry_times: int = 3, retry_delay: float = 3.0):
        """
        初始化评测器
        
        Args:
            model_name: 模型名称（如：gemini-3-pro-preview, gpt-5-mini, deepseek-reasoner等）
            concurrency: 并发数（1表示串行执行）
            reasoning_effort: 推理强度（"low", "medium", "high"，默认："low"）
            max_tokens: 最大生成token数（默认：256）
            retry_times: 重试次数（默认：3）
            retry_delay: 每次重试等待时间，单位秒（默认：3.0）
        """
        self.model_name = model_name
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        
        print(f"✓ 初始化LLM评测器")
        print(f"  模型: {model_name}")
        print(f"  并发数: {concurrency}")
        print(f"  推理强度: {reasoning_effort}")
        print(f"  最大tokens: {max_tokens}")
        print(f"  重试次数: {retry_times}")
        print(f"  重试等待: {retry_delay}秒")
    
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
    
    async def generate_async(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """
        真正的异步生成响应（使用统一llm库的acompletion，带重试机制）
        
        Args:
            prompt: 文本提示
            images: 图片路径列表
            
        Returns:
            包含响应和元数据的字典
        """
        # 构建messages
        messages = self._build_messages(prompt, images)
        
        # 实现重试逻辑
        last_error = None
        for attempt in range(self.retry_times):
            try:
                response = await acompletion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    max_tokens=self.max_tokens
                )
                
                # 成功返回
                return {
                    'success': True,
                    'response': response,
                    'error': None,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                last_error = e
                print(f"⚠️  API调用失败 (尝试 {attempt + 1}/{self.retry_times}): {str(e)}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.retry_times - 1:
                    print(f"   等待 {self.retry_delay} 秒后重试...")
                    await asyncio.sleep(self.retry_delay)
        
        # 所有重试都失败
        print(f"❌ API调用最终失败，已重试 {self.retry_times} 次")
        return {
            'success': False,
            'response': None,
            'error': str(last_error),
            'attempts': self.retry_times
        }
    
    def generate_sync(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """
        同步生成响应（使用统一llm库，带重试机制）
        
        Args:
            prompt: 文本提示
            images: 图片路径列表
            
        Returns:
            包含响应和元数据的字典
        """
        # 构建messages
        messages = self._build_messages(prompt, images)
        
        # 实现重试逻辑
        last_error = None
        for attempt in range(self.retry_times):
            try:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    max_tokens=self.max_tokens
                )
                
                # 成功返回
                return {
                    'success': True,
                    'response': response,
                    'error': None,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                last_error = e
                print(f"⚠️  API调用失败 (尝试 {attempt + 1}/{self.retry_times}): {str(e)}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.retry_times - 1:
                    print(f"   等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
        
        # 所有重试都失败
        print(f"❌ API调用最终失败，已重试 {self.retry_times} 次")
        return {
            'success': False,
            'response': None,
            'error': str(last_error),
            'attempts': self.retry_times
        }
    
    async def evaluate_batch_async(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        异步批量评测
        
        Args:
            samples: 样本列表
            
        Returns:
            评测结果列表
        """
        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # 创建进度条
        pbar = tqdm(total=len(samples), desc="评测进度", unit="样本")
        
        async def process_sample(sample, idx):
            async with semaphore:
                prompt = sample.get('prompt', '')
                images = sample.get('images', [])
                meta = sample.get('meta', {})
                
                # 调用模型（带重试）
                result = await self.generate_async(prompt, images)
                
                # 更新进度条
                pbar.update(1)

                # 返回结果
                if not result.get('success', False):
                    return {
                        'prompt': prompt,
                        'images': images,
                        'error': result.get('error'),
                        'meta': meta
                    }
                else:
                    return {
                        'prompt': prompt,
                        'images': images,
                        'response': result.get('response'),
                        'meta': meta
                    }
        
        # 创建所有任务
        tasks = [process_sample(sample, i) for i, sample in enumerate(samples)]
        
        # 并发执行，收集结果
        results = await asyncio.gather(*tasks)
        
        # 关闭进度条
        pbar.close()
        
        return results
    
    def evaluate_batch_sync(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        同步批量评测（串行执行）
        
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
            result = self.generate_sync(prompt, images)

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
    
    async def evaluate_dataset_async(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        异步评测数据集
        
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
            results = self.evaluate_batch_sync(samples)
        else:
            print(f"使用并发执行模式（并发数: {self.concurrency}）...")
            results = await self.evaluate_batch_async(samples)
        
        # 统计
        success_count = sum(1 for r in results if "response" in r)
        fail_count = len(results) - success_count
        
        print(f"\n评测完成!")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        
        return results
    
    def evaluate_dataset_sync(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        同步评测数据集（包装异步函数）
        
        Args:
            dataset_path: 数据集JSONL文件路径
            max_samples: 最大样本数（用于调试）
            
        Returns:
            评测结果列表
        """
        if self.concurrency == 1:
            # 串行执行，直接调用同步方法
            return asyncio.run(self.evaluate_dataset_async(dataset_path, max_samples))
        else:
            # 并发执行，使用asyncio
            return asyncio.run(self.evaluate_dataset_async(dataset_path, max_samples))
    
    async def retry_errors_async(self, result_path: str) -> List[Dict[str, Any]]:
        """
        重新评测包含错误的样本
        
        Args:
            result_path: 已有的评测结果文件路径
            
        Returns:
            更新后的评测结果列表
        """
        print(f"\n开始重新评测错误样本: {Path(result_path).name}")
        
        # 加载现有结果
        all_results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                all_results.append(result)
        
        print(f"加载了 {len(all_results)} 个结果")
        
        # 找出包含错误的样本及其索引
        error_samples = []
        error_indices = []
        
        for idx, result in enumerate(all_results):
            if 'error' in result and 'response' not in result:
                # 构建样本格式（用于重新评测）
                sample = {
                    'prompt': result.get('prompt', ''),
                    'images': result.get('images', []),
                    'meta': result.get('meta', {})
                }
                error_samples.append(sample)
                error_indices.append(idx)
        
        if not error_samples:
            print("没有找到包含错误的样本，无需重新评测")
            return all_results
        
        print(f"找到 {len(error_samples)} 个错误样本，开始重新评测...")
        
        # 重新评测错误样本
        if self.concurrency == 1:
            print("使用串行执行模式...")
            new_results = self.evaluate_batch_sync(error_samples)
        else:
            print(f"使用并发执行模式（并发数: {self.concurrency}）...")
            new_results = await self.evaluate_batch_async(error_samples)
        
        # 将新结果替换回原位置
        for idx, new_result in zip(error_indices, new_results):
            all_results[idx] = new_result
        
        # 统计重新评测结果
        retry_success = sum(1 for r in new_results if "response" in r)
        retry_fail = len(new_results) - retry_success
        
        print(f"\n重新评测完成!")
        print(f"  重试样本数: {len(error_samples)}")
        print(f"  本次成功: {retry_success}")
        print(f"  仍然失败: {retry_fail}")
        
        # 统计总体结果
        total_success = sum(1 for r in all_results if "response" in r)
        total_fail = len(all_results) - total_success
        
        print(f"\n总体统计:")
        print(f"  总样本数: {len(all_results)}")
        print(f"  成功: {total_success}")
        print(f"  失败: {total_fail}")
        
        return all_results
    
    def retry_errors_sync(self, result_path: str) -> List[Dict[str, Any]]:
        """
        同步重新评测包含错误的样本（包装异步函数）
        
        Args:
            result_path: 已有的评测结果文件路径
            
        Returns:
            更新后的评测结果列表
        """
        return asyncio.run(self.retry_errors_async(result_path))
    
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
    parser.add_argument('--retry-times', type=int, default=1,
                       help='API调用失败时的重试次数（默认: 3）')
    parser.add_argument('--retry-delay', type=float, default=3.0,
                       help='每次重试前的等待时间，单位秒（默认: 3.0）')
    parser.add_argument('--processed-root', type=str,
                       default='/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/processed',
                       help='处理后数据根目录')
    parser.add_argument('--output-root', type=str,
                       default='/data/data-pool/dingyifan/GeminiEvaluation/workspace/results',
                       help='结果输出根目录')
    parser.add_argument('--retry-errors', type=str, default=None,
                       help='重新评测错误样本的结果文件路径（如: /path/to/results.jsonl）')
    
    args = parser.parse_args()
    
    # 加载环境变量（根据模型自动选择）
    env_path = Path('/data/data-pool/dingyifan/GeminiEvaluation/.env')
    if env_path.exists():
        load_dotenv(env_path)
    
    # 创建评测器
    evaluator = LLMEvaluator(
        model_name=args.model,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        retry_times=args.retry_times,
        retry_delay=args.retry_delay
    )
    
    # 判断是重新评测错误还是正常评测
    if args.retry_errors:
        # 重新评测错误样本模式
        result_path = Path(args.retry_errors)
        
        if not result_path.exists():
            print(f"错误: 找不到结果文件: {result_path}")
            return
        
        print(f"\n{'='*80}")
        print("重新评测错误样本模式")
        print(f"{'='*80}")
        
        # 执行重新评测
        start_time = time.time()
        results = evaluator.retry_errors_sync(str(result_path))
        end_time = time.time()
        
        # 保存更新后的结果（覆盖原文件）
        # 先备份原文件
        backup_path = result_path.parent / f"{result_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        import shutil
        shutil.copy2(result_path, backup_path)
        print(f"\n原文件已备份到: {backup_path}")
        
        # 保存更新后的结果
        evaluator.save_results(results, str(result_path))
        
        # 打印统计信息
        print(f"\n{'='*80}")
        print("重新评测统计")
        print(f"{'='*80}")
        print(f"结果文件: {result_path.name}")
        print(f"总样本数: {len(results)}")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        print(f"{'='*80}")
        
    else:
        # 正常评测模式
        # 构建数据集路径
        dataset_path = Path(args.processed_root) / f"{args.dataset}.jsonl"
        
        if not dataset_path.exists():
            print(f"错误: 找不到数据集文件: {dataset_path}")
            print(f"\n可用的数据集:")
            for jsonl_file in sorted(Path(args.processed_root).glob('*.jsonl')):
                print(f"  - {jsonl_file.stem}")
            return
        
        # 执行评测
        start_time = time.time()
        results = evaluator.evaluate_dataset_sync(
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

