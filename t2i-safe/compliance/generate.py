import os
import json
import argparse

from client import Client

def valid_prompt(prompt):
    refused_words = ["I am unable", "I cannot"]
    if prompt is None:
        return False
    if len(prompt.strip()) == 0:
        return False
    if any(refused_word in prompt for refused_word in refused_words):
        return False
    return True

def build_prompt(item):
    """
    Constructs the prompt.
    
    [Note]: 
    The specific method for constructing these prompts is currently unpublished. 
    The implementation of this function will be updated and released after the paper is officially published.
    """
    raise NotImplementedError(
        "The prompt construction method is currently withheld prior to paper publication. "
    )
               
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini", help="Name of the model to use.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to the prompts JSON file.")
    parser.add_argument("--results_root", type=str, required=True, help="Directory to save the results.")
    return parser.parse_args()

def main(args):
    client = Client(args.model)

    results_dir = os.path.join(args.results_root, args.model)
    os.makedirs(results_dir, exist_ok=True)
    with open(args.prompts_file, "r") as f:
        data = json.load(f)
    
    json_dir = os.path.join(results_dir, "json")
    img_dir = os.path.join(results_dir, "img")
    json_path = os.path.join(json_dir, "results.json")

    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    for item in data:
        res = build_prompt(item)
        if not res:
            continue
        
        theme, description, prompt = res
        if prompt in results:
            continue
        results[prompt] = {}
        results[prompt]["theme"] = theme
        results[prompt]["description"] = description
        print(f"Generating image use prompt: {prompt}")
        rt = client.generate_image(
            prompt=prompt,
            save_dir=img_dir
        )
        results[prompt]["result"] = rt
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__": 
    args = parse_args()
    main(args)
