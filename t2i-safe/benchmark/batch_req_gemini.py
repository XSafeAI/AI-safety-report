import json
import threading
import os
from queue import Queue
from google import genai
from google.genai import types
from PIL import Image

# Configuration
API_KEY = ""
BASE_URL = ''
INPUT_FILE = 'safety_toxic.jsonl'
OUTPUT_DIR = 'output_results'
NUM_THREADS = 32

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Client
client = genai.Client(
    http_options={
        'api_version': 'v1alpha',
        'base_url': BASE_URL
    },
    api_key=API_KEY
)

def truncate_long_data(data):
    if isinstance(data, dict):
        return {k: truncate_long_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_long_data(i) for i in data]
    elif isinstance(data, bytes):
        return f"<bytes length={len(data)}>"
    elif isinstance(data, str) and len(data) > 500: 
        return data[:500] + "..."
    return data

def worker(q):
    while True:
        item = q.get()
        if item is None:
            break
        
        index, prompt_text = item
        file_id = f"{index:04d}"
        
        print(f"[{file_id}] Processing...")

        try:
            response = client.models.generate_content(
                model="gemini-3-pro-image-native",
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                        image_size="1K"
                    ),
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )

            # Check for block
            is_blocked = False
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback is not None:
                print(f"[{file_id}] BLOCKED. Feedback: {response.prompt_feedback}")
                is_blocked = True

            # Check for image parts safely
            image_part = None
            try:
                # Fix for 'NoneType' object is not iterable
                if hasattr(response, 'parts') and response.parts is not None:
                    image_parts = [part for part in response.parts if part.inline_data]
                    if image_parts:
                        image_part = image_parts[0]
            except Exception as e:
                print(f"[{file_id}] Error parsing parts: {e}")
                # Treat as no image found

            
            should_save_json = False
            
            if is_blocked:
                should_save_json = True
            elif image_part:
                should_save_json = True
            else:
                print(f"[{file_id}] No image parts found and not blocked. Skipping save.")

            if should_save_json:
                # Prepare data for JSON
                data_dict = None
                if hasattr(response, 'model_dump'):
                    data_dict = response.model_dump()
                elif hasattr(response, 'to_dict'):
                    data_dict = response.to_dict()
                
                if data_dict:
                    safe_data = truncate_long_data(data_dict)
                    # Add block status key
                    safe_data['is_blocked'] = is_blocked
                    
                    json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(safe_data, f, indent=4, ensure_ascii=False)

                # Save Image if available
                if image_part:
                    try:
                        image = image_part.as_image()
                        image.save(os.path.join(OUTPUT_DIR, f"{file_id}.png"))
                        print(f"[{file_id}] Image saved.")
                    except Exception as e:
                        print(f"[{file_id}] Error saving image: {e}")

        except Exception as e:
            print(f"[{file_id}] Exception: {e}")
            # Do not save anything on exception
        
        finally:
            q.task_done()


def main():
    # Load prompts
    prompts = []
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

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