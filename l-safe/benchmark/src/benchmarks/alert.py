import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils import get_client, generate_text, save_jsonl

def process_single(item, model_name, client):
    prompt = item.get("prompt", "")
    try:
        if not prompt:
            raise ValueError("Empty prompt")
        messages = [{"role": "user", "content": prompt}]
        response = generate_text(client, model_name, messages)
        item["response"] = response
    except Exception as e:
        item["error"] = str(e)
    return item

def run(data_path, output_path, model_name, max_workers):
    logger = logging.getLogger("ALERT")
    data_list = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    client = get_client()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(process_single, item, model_name, client) for item in data_list]
        
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="ALERT"):
            res = future.result()
            save_jsonl(res, output_path)