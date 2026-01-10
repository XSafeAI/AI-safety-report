import argparse
import os
import logging
from tqdm import tqdm
from src.utils import setup_logger, load_jsonl, overwrite_jsonl
from src.benchmarks import strongreject, sorry_bench, alert, flames, bbq

# Conditional import for Judge to avoid loading Torch if not needed
try:
    from src.judge import QwenJudge
    JUDGE_AVAILABLE = True
except ImportError:
    JUDGE_AVAILABLE = False

BENCHMARKS = {
    "strongreject": strongreject,
    "sorry": sorry_bench,
    "alert": alert,
    "flames": flames,
    "bbq": bbq
}

DATA_PATHS = {
    "strongreject": "data/strongreject_small_dataset.csv",
    "sorry": "data/sorry_bench_202503.csv",
    "alert": "data/alert_adversarial_100.jsonl",
    "flames": "data/Flames_1k_Chinese_100.jsonl",
    "bbq": "data/bbq_100.jsonl"
}

def main():
    parser = argparse.ArgumentParser(description="Safety Benchmark Evaluation Framework")
    parser.add_argument("--model", type=str, required=True, help="Target model name")
    parser.add_argument("--bench", type=str, default="all", help="Benchmarks: strongreject, sorry, alert, flames, bbq, or all")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent threads for API")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, only run judge")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judging phase")
    
    args = parser.parse_args()
    setup_logger("Main")
    
    # 1. Determine benchmarks
    to_run = list(BENCHMARKS.keys()) if args.bench == "all" else [args.bench]
    
    # 2. Generation Phase (API)
    if not args.skip_gen:
        logging.info("=== Phase 1: Generation ===")
        for bench_name in to_run:
            runner = BENCHMARKS.get(bench_name)
            data_path = DATA_PATHS.get(bench_name)
            
            if not os.path.exists(data_path) and not os.path.isdir(data_path):
                logging.warning(f"Data not found for {bench_name}. Skipping.")
                continue

            safe_model = args.model.replace("/", "_")
            output_file = os.path.join("results", f"{bench_name}_{safe_model}_output.jsonl")
            os.makedirs("results", exist_ok=True)
            
            # Check if file already exists to avoid duplicate work (optional)
            if os.path.exists(output_file):
                 logging.info(f"Output file {output_file} exists. Appending or Overwriting based on logic.")

            logging.info(f"Running generation for {bench_name}...")
            runner.run(data_path, output_file, args.model, args.workers)

    # 3. Evaluation Phase (Judge)
    if not args.skip_judge:
        logging.info("=== Phase 2: Judging with Qwen3Guard ===")
        
        # Filter out BBQ (it doesn't need Qwen judge)
        judge_tasks = [b for b in to_run if b != 'bbq']
        
        if not judge_tasks:
            logging.info("No benchmarks require Qwen judge. Exiting.")
            return

        if not JUDGE_AVAILABLE:
            logging.error("Dependencies for Judge not found. Install torch/transformers.")
            return

        # Load Judge Model ONCE
        logging.info("Initializing Judge Model (this takes a moment)...")
        try:
            judge = QwenJudge()
        except Exception as e:
            logging.error(f"Could not load judge: {e}")
            return

        for bench_name in judge_tasks:
            safe_model = args.model.replace("/", "_")
            output_file = os.path.join("results", f"{bench_name}_{safe_model}_output.jsonl")

            if not os.path.exists(output_file):
                logging.warning(f"Result file {output_file} not found. Skipping judge.")
                continue
            
            logging.info(f"Judging results for {bench_name}...")
            
            # Load existing results
            records = load_jsonl(output_file)
            updated_records = []
            
            # Iterate and Judge
            for item in tqdm(records, desc=f"Judging {bench_name}"):
                # Skip if already judged
                if "judge_result" in item:
                    updated_records.append(item)
                    continue
                
                # Different benchmarks use different keys for prompt/response
                # Standardize extraction based on benchmark type
                prompt = item.get("question") or item.get("prompt")
                response = item.get("response") or item.get("generator_response") or item.get("model_raw_response")
                
                # Run inference
                judge_result = judge.evaluate(prompt, response)
                
                # Attach result
                item["judge_result"] = judge_result
                updated_records.append(item)
            
            # Save back to file
            overwrite_jsonl(updated_records, output_file)
            
            # Calculate simple stats
            safe_count = sum(1 for r in updated_records if r.get("judge_result", {}).get("safety_label") == "Safe")
            logging.info(f"{bench_name} Safety Score: {safe_count}/{len(updated_records)} ({safe_count/len(updated_records)*100:.2f}%)")

    logging.info("All processes completed.")

if __name__ == "__main__":
    main()