import argparse
import os
import logging
from src.utils import setup_logger, load_json, save_json
from src.generation import ComplianceGenerator
from src.judge import ComplianceJudge

# Configuration from environment variables
def get_env_var(name, default=None, required=False):
    val = os.getenv(name, default)
    if required and not val:
        raise ValueError(f"Environment variable {name} is required.")
    return val

DATASETS = {
    "eu": "data/EU_AI_Act.json",
    "feat": "data/FEAT.json",
    "nist": "data/NIST.json"
}

def main():
    parser = argparse.ArgumentParser(description="LLM Compliance Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name to test")
    parser.add_argument("--dataset", type=str, default="all", help="eu, feat, nist, or all")
    parser.add_argument("--workers", type=int, default=10, help="Concurrency limit")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation, only run judge")
    
    args = parser.parse_args()
    setup_logger("Main")

    # 1. Setup Configuration
    gen_base_url = get_env_var("GENERATOR_BASE_URL", required=True)
    gen_api_key = get_env_var("GENERATOR_API_KEY", required=True)
    
    judge_base_url = get_env_var("JUDGE_BASE_URL", required=True)
    judge_api_key = get_env_var("JUDGE_API_KEY", default="EMPTY")
    judge_model = get_env_var("JUDGE_MODEL_NAME", required=True)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for key in targets:
        input_file = DATASETS[key]
        if not os.path.exists(input_file):
            logging.warning(f"Dataset not found: {input_file}. Skipping.")
            continue

        safe_model = args.model.replace("/", "_")
        output_file = os.path.join("results", f"{key}_{safe_model}_output.json")
        os.makedirs("results", exist_ok=True)
        
        data_to_judge = []

        # --- Phase 1: Generation ---
        if not args.skip_gen:
            logging.info(f"--- Starting Generation for {key.upper()} ---")
            raw_data = load_json(input_file)
            
            generator = ComplianceGenerator(
                base_url=gen_base_url,
                api_key=gen_api_key,
                model_name=args.model,
                max_workers=args.workers
            )
            data_to_judge = generator.run(raw_data)
            save_json(data_to_judge, output_file)
        else:
            if os.path.exists(output_file):
                logging.info(f"Loading existing results for {key.upper()}...")
                data_to_judge = load_json(output_file)
            else:
                logging.error(f"Output file {output_file} not found. Cannot skip generation.")
                continue

        # --- Phase 2: Judging ---
        logging.info(f"--- Starting Judging for {key.upper()} ---")
        
        judge = ComplianceJudge(
            base_url=judge_base_url,
            api_key=judge_api_key,
            model_name=judge_model,
            max_workers=args.workers
        )
        
        final_results = judge.evaluate(data_to_judge)
        save_json(final_results, output_file) # Overwrite with judged results
        
        judge.print_report(final_results, key.upper())

    logging.info("All tasks completed.")

if __name__ == "__main__":
    main()