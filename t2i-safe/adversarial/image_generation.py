import csv
import os
import base64

import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
import argparse
from google import genai
from google.genai import types
from openai import OpenAI

# Google AI API配置
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # 替换为实际的API密钥
DOUBAO_API_KEY = "YOUR_DOUBAO_API_KEY"  # 替换为实际的API密钥
# 攻击方法和领域
ATTACK_METHODS = ["pgj", "genbreak"]
CATEGORIES = ["nudity", "violence", "hate"]

# 最大重试次数
MAX_RETRIES = 1
# 当遇到 429 时的最大不计入次数的重试限制
RATE_LIMIT_MAX_RETRIES = 5
# 429 重试的基准等待时间（秒），采用指数退避
RATE_LIMIT_BACKOFF_BASE = 5

# 输出目录
OUTPUT_DIR = "generated_images"
# 日志文件模板，会按模型名生成不同文件
LOG_FILE_TEMPLATE = "evaluation_log_{model}.json"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志记录
evaluation_log = []


# 初始化 clients 字典（将为每个模型创建对应的客户端）
CLIENTS: Dict[str, object] = {}


# 支持的模型配置（可拓展）
# provider: 'google' 或 'doubao' （未来可扩展）
MODELS = [
    {
        "name": "gemini-3-pro-image",
        "provider": "google",
        "model_id": "gemini-3-pro-image-preview",
    },
    {
        "name": "doubao-seedream-4-0",
        "provider": "doubao",
        "model_id": "doubao-seedream-4-0-250828",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    {
        "name": "doubao-seedream-4-5",
        "provider": "doubao",
        "model_id": "doubao-seedream-4-5-251128",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
]


# Create clients for configured models in a consistent location
for _m in MODELS:
    name = _m["name"]
    provider = _m.get("provider")
    if provider == "google":
        CLIENTS[name] = genai.Client(api_key=GOOGLE_API_KEY)
    elif provider == "doubao":
        api_key = DOUBAO_API_KEY
        base_url = _m.get("base_url")
        CLIENTS[name] = OpenAI(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider {provider} for model {name}")



def load_prompts(method: str, category: str) -> List[str]:
    file_path = f"data/{method}_{category}.csv"
    prompts = []

    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompts.append(row["prompt"])
    prefix = "Generate a picture according to the following instruction: \n"
    prompts = [prefix + prompt for prompt in prompts]
    return prompts


def _safe_get_data_item(resp):
    if not resp:
        return None
    data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
    if not data:
        return None
    return data[0] if len(data) > 0 else None


def _is_rate_limit_exception(exc: Exception) -> bool:
    """粗略检测是否为速率限制/429 错误。
    支持多种异常类型的检测：requests 的 HTTPError、包含 response.status_code、或消息中包含 '429' / 'rate limit'。
    """
    try:
        # requests HTTPError
        import requests as _req

        if isinstance(exc, _req.exceptions.HTTPError):
            resp = getattr(exc, "response", None)
            if resp is not None and getattr(resp, "status_code", None) == 429:
                return True
    except Exception:
        pass

    # 通用检查：有 response 属性
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True

    # 有 status_code 属性
    if getattr(exc, "status_code", None) == 429:
        return True

    # 文本匹配作为兜底
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True

    return False


def save_image_from_openai_response(resp, image_path: str) -> bool:
    """Handle responses similar to `test.py`'s OpenAI-like images.generate response."""
    item = _safe_get_data_item(resp)
    if not item:
        return False

    # Try URL first
    url = None
    try:
        url = getattr(item, "url", None)
    except Exception:
        url = None
    if url is None and isinstance(item, dict):
        url = item.get("url")

    if url:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(r.content)
            return True
        except Exception as e:
            print(f"Failed downloading image from url: {e}")

    # Fallback to base64
    b64 = None
    try:
        b64 = getattr(item, "b64_json", None)
    except Exception:
        b64 = None
    if b64 is None and isinstance(item, dict):
        b64 = item.get("b64_json") or item.get("b64") or item.get("b64_image")

    if b64:
        try:
            data = base64.b64decode(b64)
            with open(image_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"Failed saving base64 image: {e}")

    return False


def generate_image_for_model(prompt: str, model_cfg: Dict, method: str, category: str, index: int) -> Dict:
    """Dispatch generation to the appropriate provider based on model_cfg."""
    model_name = model_cfg["name"]
    # get client initialized in CLIENTS
    client = CLIENTS.get(model_name)
    if client is None:
        return {
            "model": model_name,
            "method": method,
            "category": category,
            "index": index,
            "prompt": prompt,
            "status": "failed",
            "reason": "no_client",
            "image_path": None,
        }
    # 创建输出目录（包含模型名，便于区分）
    category_dir = os.path.join(OUTPUT_DIR, model_name, method, category)
    os.makedirs(category_dir, exist_ok=True)

    image_path = os.path.join(category_dir, f"{model_name}_{method}_{category}_{index}.png")
    image_abs_path = os.path.abspath(image_path)

    if os.path.exists(image_path):
        return {
            "model": model_name,
            "method": method,
            "category": category,
            "index": index,
            "prompt": prompt,
            "status": "skipped",
            "reason": "already_exists",
            "image_path": image_abs_path,
        }

    attempt = 0
    rate_limit_tries = 0
    while attempt < MAX_RETRIES:
        try:
            if model_cfg["provider"] == "google":
                response = client.models.generate_content(
                    model=model_cfg["model_id"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(aspect_ratio="1:1")
                    ),
                )
                # 保存图像
                for part in response.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        image.save(image_path)
                        return {
                            "model": model_name,
                            "method": method,
                            "category": category,
                            "index": index,
                            "prompt": prompt,
                            "status": "success",
                            "attempt": attempt + 1,
                            "image_path": image_abs_path,
                        }

                print(f"Attempt {attempt+1} failed for {model_name} prompt {index}: No image data returned")
                time.sleep(1)
                continue

            elif model_cfg["provider"] == "doubao":
                imagesResponse = client.images.generate(
                    model=model_cfg["model_id"],
                    prompt=prompt,
                    size="2048x2048",
                    response_format="url",
                    extra_body={"watermark": True},
                )

                saved = save_image_from_openai_response(imagesResponse, image_path)
                if saved:
                    return {
                        "model": model_name,
                        "method": method,
                        "category": category,
                        "index": index,
                        "prompt": prompt,
                        "status": "success",
                        "attempt": attempt + 1,
                        "image_path": image_abs_path,
                    }

                print(f"Attempt {attempt+1} failed for {model_name} prompt {index}: No image data returned")
                time.sleep(1)
                continue

            else:
                return {
                    "model": model_name,
                    "method": method,
                    "category": category,
                    "index": index,
                    "prompt": prompt,
                    "status": "failed",
                    "reason": "unsupported_provider",
                    "image_path": None,
                }

        except Exception as e:
            # 如果是 429（速率限制），则不计入常规重试次数，进行特殊退避后重试
            if _is_rate_limit_exception(e) and rate_limit_tries < RATE_LIMIT_MAX_RETRIES:
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** rate_limit_tries)
                print(f"Rate limited (429) for {model_name} prompt {index}, waiting {wait}s before retrying (not counting against MAX_RETRIES)")
                time.sleep(wait)
                rate_limit_tries += 1
                continue

            # 非 429 或超出 rate-limit retry 次数，计入常规尝试次数
            attempt += 1
            print(f"Attempt {attempt} exception for {model_name} prompt {index}: {str(e)}")
            # 常规指数退避
            try:
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(1)
            continue

    return {
        "model": model_name,
        "method": method,
        "category": category,
        "index": index,
        "prompt": prompt,
        "status": "failed",
        "attempts": MAX_RETRIES,
        "image_path": None,
    }


def process_prompts_for_model_method_category(model_cfg: Dict, method: str, category: str):
    print(f"Processing model={model_cfg['name']} {method} - {category}")
    prompts = load_prompts(method, category)
    print(f"Loaded {len(prompts)} prompts for {method} - {category}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(generate_image_for_model, prompt, model_cfg, method, category, i)
            for i, prompt in enumerate(prompts)
        ]

        for future in as_completed(futures):
            result = future.result()
            evaluation_log.append(result)
            print(f"Completed: {result.get('model')}-{result['method']}-{result['category']}-{result['index']} - Status: {result['status']}")

    # 写入按模型划分的日志文件
    model_log_file = LOG_FILE_TEMPLATE.format(model=model_cfg["name"])
    model_entries = [item for item in evaluation_log if item.get("model") == model_cfg["name"]]
    with open(model_log_file, "w", encoding="utf-8") as f:
        json.dump(model_entries, f, indent=2, ensure_ascii=False)
    print(f"Log updated ({model_log_file}) for model={model_cfg['name']} {method} - {category}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型名，必须是 MODELS 中配置的 name 字段")
    args = parser.parse_args()
    selected_model = args.model

    # 查找模型配置
    model_cfg = None
    for m in MODELS:
        if m["name"] == selected_model:
            model_cfg = m
            break

    if model_cfg is None:
        print("Unknown model:\nAvailable models:")
        for m in MODELS:
            print(" -", m["name"])
        return

    print(f"Starting Safety Evaluation for model={selected_model}")

    # 仅对所选模型进行所有 method/category 的测试
    for method in ATTACK_METHODS:
        for category in CATEGORIES:
            process_prompts_for_model_method_category(model_cfg, method, category)

    success_count = sum(1 for item in evaluation_log if item.get("status") == "success")
    failed_count = sum(1 for item in evaluation_log if item.get("status") == "failed")
    skipped_count = sum(1 for item in evaluation_log if item.get("status") == "skipped")

    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total processed: {len(evaluation_log)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
