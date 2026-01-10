import os
import logging
import time
import json
from functools import wraps
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration defaults
DEFAULT_MAX_WORKERS = 10
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 1

def load_jsonl(path):
    """Reads a JSONL file into a list of dictionaries."""
    data = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def overwrite_jsonl(data, output_path):
    """Overwrites a JSONL file with the provided list of data."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def setup_logger(name, log_file=None):
    """Sets up a logger that outputs to console and optionally to a file."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger

def get_client(base_url=None, api_key=None):
    """Initializes OpenAI client using args or env vars."""
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY")
    )

def retry_with_backoff(attempts=DEFAULT_RETRIES, initial_delay=DEFAULT_RETRY_DELAY):
    """Decorator for exponential backoff retries."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < attempts - 1:
                        print(f"[WARN] {func.__name__} failed (Attempt {i+1}/{attempts}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        print(f"[ERROR] {func.__name__} failed after {attempts} attempts.")
                        raise e
        return wrapper
    return decorator

@retry_with_backoff()
def generate_text(client, model, messages, temperature=0.7):
    """Generic generation function."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    try:
        return response.choices[0].message.reasoning_content
    except (AttributeError, TypeError):
        return response.choices[0].message.content

def save_jsonl(data, output_path, mode='a'):
    """Helper to append a single dict to a JSONL file."""
    with open(output_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')