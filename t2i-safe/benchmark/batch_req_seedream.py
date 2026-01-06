import json
import threading
import os
from queue import Queue

import io
import time
import base64
import requests
from PIL import Image
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    print("Error: The 'volcenginesdkarkruntime' library is not installed.")
    print("Please install it by running: pip install 'volcengine-python-sdk[ark]'")
    exit(1)

# Configuration
# Endpoint ID ("ep-2024xxxx-xxxxx")
MODEL_ID = "" 
API_KEY = ""  
INPUT_FILE = 'safety_toxic.jsonl'
OUTPUT_DIR = 'output_results_doubao'
NUM_THREADS = 8  # QPS


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


    

def convert_image_to_base64(image_input: Image.Image | bytes | bytearray) -> str:
    if isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        img_byte_arr = io.BytesIO()

        image_input.save(img_byte_arr, format='PNG')
        img_binary_data = img_byte_arr.getvalue()
        
    elif isinstance(image_input, (bytes, bytearray)):
        img_binary_data = image_input
    else:
        raise ValueError('Not supported')

    base64_encoded_data = base64.b64encode(img_binary_data).decode('utf-8')
    return f"data:image/png;base64,{base64_encoded_data}"



class Seedream:
    def __init__(self, model, use_watermark: bool=True, guidance_scale: float=5.5, max_retries: int=5):
        self.model = model
        self.url = "https://ark.cn-beijing.volces.com/api/v3"
        self.api_key = API_KEY
        self.use_watermark = use_watermark
        self.guidance_scale = guidance_scale
        
    def generate(self, prompt: str, size: str='1k') -> Image.Image:
        client = Ark(
            base_url=self.url,
            api_key=self.api_key
        )

        for _ in range(self.max_retries):
            try:
                response = client.images.generate(
                    model=self.model,
                    prompt=prompt,
                    response_format="url",
                    # size=size,
                    guidance_scale=self.guidance_scale,
                    watermark=self.use_watermark
                )

                image_url = response.data[0].url
                
                generated_image = Image.open(io.BytesIO(requests.get(image_url).content))
                return generated_image

            except Exception as e:
                print(f"发生错误: {e}")
                print("等待5秒后重试...")
                time.sleep(5)
                continue

    def edit(self, prompt: str, image: Image.Image, size='adaptive') -> Image.Image:
        client = Ark(
            base_url=self.url,
            api_key=self.api_key
        )

        for _ in range(self.max_retries):
            try:
                response = client.images.generate(
                    model=self.model,
                    prompt=prompt,
                    image=convert_image_to_base64(image), 
                    response_format="url",
                    # size=size,
                    guidance_scale=self.guidance_scale,
                    watermark=self.use_watermark
                )

                image_url = response.data[0].url
                
                edited_image = Image.open(io.BytesIO(requests.get(image_url).content))
                return edited_image

            except Exception as e:
                # 捕获异常信息
                error_message = str(e)
                print(f"发生错误: {e}")
                # 判断是否是特定的错误类型：如内容被禁止的错误
                if "InputImageSensitiveContentDetected" in error_message:
                    print("检测到输入图像敏感内容，跳过此指令。")
                    break 
                else:
                    # 其他错误，进行重试
                    print("等待5秒后重试...")
                    time.sleep(5)
                    continue
        



def worker(q):
    # 初始化模型实例
    # 注意：Seedream 类内部处理了鉴权，这里只需要传入 Endpoint ID
    try:
        model_client = Seedream(model=MODEL_ID)
        if not hasattr(model_client, 'max_retries'):
            model_client.max_retries = 1
    except Exception as e:
        print(f"Error initializing model client: {e}")
        return

    while True:
        item = q.get()
        if item is None:
            break
        
        index, prompt_text = item
        file_id = f"{index:04d}"
        
        print(f"[{file_id}] Processing...")

        try:
            image = model_client.generate(prompt=prompt_text)

            result_info = {
                "file_id": file_id,
                "prompt": prompt_text,
                "status": "unknown",
                "error": None
            }

            if image:
                image_filename = f"{file_id}.png"
                image_path = os.path.join(OUTPUT_DIR, image_filename)
                image.save(image_path)
                print(f"[{file_id}] Image saved.")
                
                result_info["status"] = "success"
                result_info["image_path"] = image_filename
            else:
                print(f"[{file_id}] Generation failed (No image returned).")
                result_info["status"] = "failed"
                result_info["error"] = "Max retries exceeded or API error"


            json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_info, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[{file_id}] Exception: {e}")

            error_info = {
                "file_id": file_id,
                "prompt": prompt_text,
                "status": "error",
                "error": str(e)
            }
            json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=4, ensure_ascii=False)
        
        finally:
            q.task_done()

def main():
    # Load prompts
    prompts = []
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading prompts from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        prompts.append(data["prompt"])
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(prompts)} prompts. Starting {NUM_THREADS} threads...")

    q = Queue()
    threads = []

    # Start threads
    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(q,))
        t.start()
        threads.append(t)

    # Enqueue jobs
    for i, prompt in enumerate(prompts):
        q.put((i, prompt))

    # Wait for queue to empty
    q.join()

    # Stop threads
    for _ in range(NUM_THREADS):
        q.put(None)
    for t in threads:
        t.join()

    print("Batch processing complete.")

if __name__ == "__main__":
    main()