"""
LLM model evaluation script
Uses unified llm library for inference evaluation on datasets
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

# Import unified llm library
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import completion, acompletion


class LLMEvaluator:
    """LLM evaluator (supports multiple models)"""
    
    def __init__(self, model_name: str, concurrency: int = 5, 
                 reasoning_effort: str = "low", max_tokens: int = 256,
                 retry_times: int = 3, retry_delay: float = 3.0):
        """
        Initialize evaluator
        
        Args:
            model_name: Model name (e.g.: gemini-3-pro-preview, gpt-5-mini, deepseek-reasoner, etc.)
            concurrency: Concurrency level (1 means serial execution)
            reasoning_effort: Reasoning effort ("low", "medium", "high", default: "low")
            max_tokens: Maximum number of tokens to generate (default: 256)
            retry_times: Number of retries (default: 3)
            retry_delay: Wait time before each retry in seconds (default: 3.0)
        """
        self.model_name = model_name
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        
        print(f"✓ Initializing LLM evaluator")
        print(f"  Model: {model_name}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Retry times: {retry_times}")
        print(f"  Retry delay: {retry_delay} seconds")
    
    def _build_messages(self, prompt: str, images: List[str]) -> List[Dict[str, Any]]:
        """
        Build OpenAI-style messages format
        
        Args:
            prompt: Text prompt
            images: List of image paths
            
        Returns:
            List of messages
        """
        # If no images, return text message directly
        if not images:
            return [{"role": "user", "content": prompt}]
        
        # Build multimodal content
        content = []
        
        # Add images
        for img_path in images:
            if os.path.exists(img_path):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_path  # Local path, will be automatically processed
                    }
                })
        
        # Add text
        content.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content}]
    
    async def generate_async(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """
        True async response generation (using unified llm library's acompletion with retry mechanism)
        
        Args:
            prompt: Text prompt
            images: List of image paths
            
        Returns:
            Dictionary containing response and metadata
        """
        # Build messages
        messages = self._build_messages(prompt, images)
        
        # Implement retry logic
        last_error = None
        for attempt in range(self.retry_times):
            try:
                response = await acompletion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    max_tokens=self.max_tokens
                )
                
                # Return on success
                return {
                    'success': True,
                    'response': response,
                    'error': None,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                last_error = e
                print(f"⚠️  API call failed (attempt {attempt + 1}/{self.retry_times}): {str(e)}")
                
                # If not last attempt, wait before retry
                if attempt < self.retry_times - 1:
                    print(f"   Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        print(f"❌ API call finally failed after {self.retry_times} retries")
        return {
            'success': False,
            'response': None,
            'error': str(last_error),
            'attempts': self.retry_times
        }
    
    def generate_sync(self, prompt: str, images: List[str]) -> Dict[str, Any]:
        """
        Synchronous response generation (using unified llm library with retry mechanism)
        
        Args:
            prompt: Text prompt
            images: List of image paths
            
        Returns:
            Dictionary containing response and metadata
        """
        # Build messages
        messages = self._build_messages(prompt, images)
        
        # Implement retry logic
        last_error = None
        for attempt in range(self.retry_times):
            try:
                response = completion(
                    model=self.model_name,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    max_tokens=self.max_tokens
                )
                
                # Return on success
                return {
                    'success': True,
                    'response': response,
                    'error': None,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                last_error = e
                print(f"⚠️  API call failed (attempt {attempt + 1}/{self.retry_times}): {str(e)}")
                
                # If not last attempt, wait before retry
                if attempt < self.retry_times - 1:
                    print(f"   Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        print(f"❌ API call finally failed after {self.retry_times} retries")
        return {
            'success': False,
            'response': None,
            'error': str(last_error),
            'attempts': self.retry_times
        }
    
    async def evaluate_batch_async(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Async batch evaluation
        
        Args:
            samples: List of samples
            
        Returns:
            List of evaluation results
        """
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Create progress bar
        pbar = tqdm(total=len(samples), desc="Evaluation progress", unit="samples")
        
        async def process_sample(sample, idx):
            async with semaphore:
                prompt = sample.get('prompt', '')
                images = sample.get('images', [])
                meta = sample.get('meta', {})
                
                # Call model (with retry)
                result = await self.generate_async(prompt, images)
                
                # Update progress bar
                pbar.update(1)

                # Return result
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
        
        # Create all tasks
        tasks = [process_sample(sample, i) for i, sample in enumerate(samples)]
        
        # Execute concurrently and collect results
        results = await asyncio.gather(*tasks)
        
        # Close progress bar
        pbar.close()
        
        return results
    
    def evaluate_batch_sync(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Synchronous batch evaluation (serial execution)
        
        Args:
            samples: List of samples
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Use tqdm to show progress
        for sample in tqdm(samples, desc="Evaluation progress", unit="samples"):
            prompt = sample.get('prompt', '')
            images = sample.get('images', [])
            meta = sample.get('meta', {})
            
            # Call model (with retry)
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
        Async dataset evaluation
        
        Args:
            dataset_path: Dataset JSONL file path
            max_samples: Maximum number of samples (for debugging)
            
        Returns:
            List of evaluation results
        """
        print(f"\nStarting evaluation on dataset: {Path(dataset_path).stem}")
        
        # Load dataset
        samples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                sample = json.loads(line)
                samples.append(sample)
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        print(f"Loaded {len(samples)} samples")
        
        # Evaluate
        if self.concurrency == 1:
            print("Using serial execution mode...")
            results = self.evaluate_batch_sync(samples)
        else:
            print(f"Using concurrent execution mode (concurrency: {self.concurrency})...")
            results = await self.evaluate_batch_async(samples)
        
        # Statistics
        success_count = sum(1 for r in results if "response" in r)
        fail_count = len(results) - success_count
        
        print(f"\nEvaluation completed!")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")
        
        return results
    
    def evaluate_dataset_sync(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Synchronous dataset evaluation (wraps async function)
        
        Args:
            dataset_path: Dataset JSONL file path
            max_samples: Maximum number of samples (for debugging)
            
        Returns:
            List of evaluation results
        """
        if self.concurrency == 1:
            # Serial execution, call sync method directly
            return asyncio.run(self.evaluate_dataset_async(dataset_path, max_samples))
        else:
            # Concurrent execution, use asyncio
            return asyncio.run(self.evaluate_dataset_async(dataset_path, max_samples))
    
    async def retry_errors_async(self, result_path: str) -> List[Dict[str, Any]]:
        """
        Re-evaluate samples containing errors
        
        Args:
            result_path: Path to existing evaluation result file
            
        Returns:
            Updated list of evaluation results
        """
        print(f"\nStarting re-evaluation of error samples: {Path(result_path).name}")
        
        # Load existing results
        all_results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                all_results.append(result)
        
        print(f"Loaded {len(all_results)} results")
        
        # Find samples containing errors and their indices
        error_samples = []
        error_indices = []
        
        for idx, result in enumerate(all_results):
            if 'error' in result and 'response' not in result:
                # Build sample format (for re-evaluation)
                sample = {
                    'prompt': result.get('prompt', ''),
                    'images': result.get('images', []),
                    'meta': result.get('meta', {})
                }
                error_samples.append(sample)
                error_indices.append(idx)
        
        if not error_samples:
            print("No error samples found, no need to re-evaluate")
            return all_results
        
        print(f"Found {len(error_samples)} error samples, starting re-evaluation...")
        
        # Re-evaluate error samples
        if self.concurrency == 1:
            print("Using serial execution mode...")
            new_results = self.evaluate_batch_sync(error_samples)
        else:
            print(f"Using concurrent execution mode (concurrency: {self.concurrency})...")
            new_results = await self.evaluate_batch_async(error_samples)
        
        # Replace results back to original positions
        for idx, new_result in zip(error_indices, new_results):
            all_results[idx] = new_result
        
        # Statistics for re-evaluation results
        retry_success = sum(1 for r in new_results if "response" in r)
        retry_fail = len(new_results) - retry_success
        
        print(f"\nRe-evaluation completed!")
        print(f"  Retry samples: {len(error_samples)}")
        print(f"  This round success: {retry_success}")
        print(f"  Still failed: {retry_fail}")
        
        # Overall statistics
        total_success = sum(1 for r in all_results if "response" in r)
        total_fail = len(all_results) - total_success
        
        print(f"\nOverall statistics:")
        print(f"  Total samples: {len(all_results)}")
        print(f"  Success: {total_success}")
        print(f"  Failed: {total_fail}")
        
        return all_results
    
    def retry_errors_sync(self, result_path: str) -> List[Dict[str, Any]]:
        """
        Synchronous re-evaluation of samples containing errors (wraps async function)
        
        Args:
            result_path: Path to existing evaluation result file
            
        Returns:
            Updated list of evaluation results
        """
        return asyncio.run(self.retry_errors_async(result_path))
    
    @staticmethod
    def save_results(results: List[Dict[str, Any]], output_path: str):
        """
        Save evaluation results
        
        Args:
            results: List of evaluation results
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LLM model evaluation script (supports multiple models)')
    parser.add_argument('--dataset', type=str, 
                       default='vljailbreakbench',
                       help='Dataset name (e.g.: vljailbreakbench, usb, mis_test, etc.)')
    parser.add_argument('--max-samples', type=int, default=2,
                       help='Maximum number of samples to evaluate (for debugging, default evaluates all)')
    parser.add_argument('--concurrency', type=int, default=2,
                       help='Concurrency level (set to 1 for serial execution, default 2)')
    parser.add_argument('--model', type=str, default='gemini-3-pro-preview',
                       help='Model name (supports: gemini-3-pro-preview, gemini-2.5-pro, gpt-5-mini, deepseek-reasoner, etc.)')
    parser.add_argument('--reasoning-effort', type=str, default='low',
                       choices=['low', 'medium', 'high'],
                       help='Reasoning effort (low/medium/high, default: low)')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate (default: 256)')
    parser.add_argument('--retry-times', type=int, default=1,
                       help='Number of retries on API call failure (default: 3)')
    parser.add_argument('--retry-delay', type=float, default=3.0,
                       help='Wait time before each retry in seconds (default: 3.0)')
    parser.add_argument('--processed-root', type=str,
                       default='/data/data-pool/dingyifan/VL-Safe/workspace/data/processed',
                       help='Processed data root directory')
    parser.add_argument('--output-root', type=str,
                       default='/data/data-pool/dingyifan/VL-Safe/workspace/results',
                       help='Result output root directory')
    parser.add_argument('--retry-errors', type=str, default=None,
                       help='Path to result file for re-evaluating error samples (e.g.: /path/to/results.jsonl)')
    
    args = parser.parse_args()
    
    # Load environment variables (auto select based on model)
    env_path = Path('/data/data-pool/dingyifan/VL-Safe/.env')
    if env_path.exists():
        load_dotenv(env_path)
    
    # Create evaluator
    evaluator = LLMEvaluator(
        model_name=args.model,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        retry_times=args.retry_times,
        retry_delay=args.retry_delay
    )
    
    # Determine whether to retry errors or normal evaluation
    if args.retry_errors:
        # Re-evaluation of error samples mode
        result_path = Path(args.retry_errors)
        
        if not result_path.exists():
            print(f"Error: Cannot find result file: {result_path}")
            return
        
        print(f"\n{'='*80}")
        print("Re-evaluation of Error Samples Mode")
        print(f"{'='*80}")
        
        # Execute re-evaluation
        start_time = time.time()
        results = evaluator.retry_errors_sync(str(result_path))
        end_time = time.time()
        
        # Save updated results (overwrite original file)
        # Backup original file first
        backup_path = result_path.parent / f"{result_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        import shutil
        shutil.copy2(result_path, backup_path)
        print(f"\nOriginal file backed up to: {backup_path}")
        
        # Save updated results
        evaluator.save_results(results, str(result_path))
        
        # Print statistics
        print(f"\n{'='*80}")
        print("Re-evaluation Statistics")
        print(f"{'='*80}")
        print(f"Result file: {result_path.name}")
        print(f"Total samples: {len(results)}")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        print(f"{'='*80}")
        
    else:
        # Normal evaluation mode
        # Build dataset path
        dataset_path = Path(args.processed_root) / f"{args.dataset}.jsonl"
        
        if not dataset_path.exists():
            print(f"Error: Cannot find dataset file: {dataset_path}")
            print(f"\nAvailable datasets:")
            for jsonl_file in sorted(Path(args.processed_root).glob('*.jsonl')):
                print(f"  - {jsonl_file.stem}")
            return
        
        # Execute evaluation
        start_time = time.time()
        results = evaluator.evaluate_dataset_sync(
            str(dataset_path),
            max_samples=args.max_samples
        )
        end_time = time.time()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{args.dataset}_{timestamp}.jsonl"
        output_path = Path(args.output_root) / output_filename
        
        evaluator.save_results(results, str(output_path))
        
        # Print statistics
        print(f"\n{'='*80}")
        print("Evaluation Statistics")
        print(f"{'='*80}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {len(results)}")
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Average per sample: {(end_time - start_time) / len(results):.2f} seconds")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()

