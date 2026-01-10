import os
import time
import yaml
from PIL import Image
from pathlib import Path
from openai import OpenAI  
from google import genai

from utils.file_util import convert_base64_to_image, convert_image_to_base64


class Client:
    max_retries : int = 5
    retry_delay : int = 2
    
    def __init__(
        self,
        model: str,
        config_path: str = "config/config.yaml"
    ):
        config = self.load_config(config_path)
        self.model_name = config[model]["model_name"]

        if self.model_name.startswith("gemini"):
            self.client = genai.Client(api_key=config[model]["api_key"])
        else:
            self.client = OpenAI(
                base_url=config[model]["base_url"],
                api_key=config[model]["api_key"]
            )
        
    def load_config(
        self, 
        config_path: str
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def generate_image(
        self,
        prompt: str,
        save_dir: str = None
    ) -> Image.Image:
        save_dir = Path(save_dir) if save_dir else Path.cwd()
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.max_retries):
            try:
                print(f"[INFO] Attempt {i+1} with model {self.model_name}")
                if self.model_name.startswith("gemini"):
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt]
                    )
                    output_image = [part for part in response.parts if part.inline_data is not None][0].as_image()
                else:
                    response = self.client.images.generate(
                        model=self.model_name,
                        prompt=prompt,
                        response_format="b64_json",
                    )
                    output_image = convert_base64_to_image(response.data[0].b64_json)
                save_path = os.path.join(save_dir, f"{int(time.time())}.png")
                output_image.save(save_path)
                return save_path
                
            except Exception as e:
                print(f"[ERROR] {e}")
                if "OutputImageSensitiveContentDetected" in str(e):
                    print("Detect output image sensitive content and skip this request.")
                    return "REJECTED"
                
                print(f"[INFO] Retrying after {self.retry_delay} seconds.")
                time.sleep(self.retry_delay)
                
        print(f"[ERROR] All {self.max_retries} attempts failed with model {self.model}.")
        return "FAILED"
    
    def generate_text(
        self,
        image: Image,
        prompt: str,
    ) -> Image.Image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url","image_url": {"url": convert_image_to_base64(image)}},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
                    
        for i in range(self.max_retries):
            try:
                print(f"[INFO] Attempt {i+1} with model {self.model_name}")
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"[ERROR] {e}")
                print(f"[INFO] Retrying after {self.retry_delay} seconds.")
                time.sleep(self.retry_delay)
                
        print(f"[ERROR] All {self.max_retries} attempts failed with model {self.model_name}.")
        return "FAILED"