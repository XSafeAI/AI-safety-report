import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils import get_client, generate_text, save_jsonl

def process_single(item, model_name, client):
    idx, question = item
    result = {
        "id": idx,
        "question": question,
        "model": model_name,
        "response": None,
        "error": None
    }
    try:
        messages = [{"role": "user", "content": question}]
        response = generate_text(client, model_name, messages)
        result["response"] = response
    except Exception as e:
        result["error"] = str(e)
    return result

def run(data_path, output_path, model_name, max_workers):
    logger = logging.getLogger("SorryBench")
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        # Assuming question is in column index 1 based on original code
        questions = df.iloc[:, 1].dropna().tolist()
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    client = get_client()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(process_single, (i, q), model_name, client) for i, q in enumerate(questions)]
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="SorryBench"):
            res = future.result()
            save_jsonl(res, output_path)