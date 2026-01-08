# LLM Unified Calling Library

A unified calling library supporting multiple large language models, providing a simple and consistent API interface.

## ✨ Core Features

- 🚀 **Unified Interface**: All models use the same `completion()` method
- 🔄 **Auto Routing**: Automatically selects the corresponding provider based on model name
- 🧠 **Thinking Support**: Unified thinking parameter interface, automatically handles differences between models
- 🔁 **Auto Retry**: Built-in retry mechanism to improve stability
- 💡 **Lazy Loading**: Providers initialized on demand, saving resources
- 🎯 **Simple Return**: Returns parsed text by default, optionally returns full response

## 📦 Supported Models

### 1. OpenAI (OpenAIProvider)
- **Models**: gpt-5, gpt-5-mini, gpt-4.1, gpt-4o, gpt-4o-mini
- **Thinking Support**: gpt-5 series supports reasoning_effort (low/medium/high)
- **Environment Variable**: `OPENAI_API_KEY`

### 2. Ark/Volcano Engine Doubao (ArkProvider)
- **Models**: doubao-seed-1-6-251015, doubao-seed-1-6-vision-250815
- **Thinking Support**: 
  - doubao-seed-1-6-251015: Supports reasoning_effort levels
  - doubao-seed-1-6-vision-250815: Supports extra_body thinking toggle
- **Environment Variable**: `ARK_API_KEY`
- **Feature**: Returns thinking content

### 3. DashScope/Qwen (DashScopeProvider)
- **Models**: 
  - Thinking only: qwen3-vl-*-thinking series
  - No thinking: qwen3-vl-*-instruct series
  - Flexible models: qwen2.5-vl/qwen2.5 series
- **Thinking Support**: 
  - Thinking models automatically use streaming API
  - Flexible models can be controlled via reasoning_effort
- **Environment Variable**: `DASHSCOPE_API_KEY`
- **Feature**: Automatically handles streaming responses

### 4. Gemini (GeminiProvider)
- **Models**: 
  - Gemini 3: gemini-3-pro-preview (uses thinkingLevel)
  - Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash (uses thinkingBudget)
- **Thinking Support**: 
  - Gemini 3: low/high levels
  - Gemini 2.5 Pro: Cannot be completely disabled (minimum 128)
  - Gemini 2.5 Flash: Can be disabled (set to 0)
- **Environment Variable**: `GEMINI_API_KEY`
- **Feature**: Returns thinking summary

### 5. DeepSeek (DeepSeekProvider)
- **Models**: 
  - deepseek-reasoner: Supports thinking
  - deepseek-chat: Does not support thinking
- **Thinking Support**: 
  - deepseek-reasoner automatically thinks, cannot be controlled
  - reasoning_effort parameter has no effect
- **Multimodal Support**: ❌ Does not support image and video input
- **Environment Variable**: `DEEPSEEK_API_KEY`
- **Feature**: Thinking behavior built into the model, only supports text input

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

**System Dependencies**:
- If you need to use video features (compression, frame extraction, etc.), you need to install ffmpeg:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Windows
  # Download and install from https://ffmpeg.org/download.html
  ```

### Set Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export ARK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

### Basic Usage

```python
from llm import completion

# Default returns text (recommended)
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
print(content)  # Print text directly
```

### Using Thinking Feature

```python
# OpenAI
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "What is quantum computing?"}],
    reasoning_effort="high"
)

# Ark (returns dictionary)
result = completion(
    model="doubao-seed-1-6-251015",
    messages=[{"role": "user", "content": "Complex question"}],
    reasoning_effort="medium"
)
print(result["content"])           # Answer
print(result["thinking_content"])  # Thinking process

# DeepSeek (automatic thinking)
result = completion(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Which is bigger, 9.11 or 9.8?"}]
)
print(result["content"])           # Answer
print(result["thinking_content"])  # Thinking process
```

### Return Full Response

```python
response = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}],
    return_full_response=True
)
print(response.usage)   # Token usage
print(response.model)   # Model information
```

### Auto Retry

```python
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}],
    retry_times=3,      # Retry 3 times
    retry_delay=2.0     # Wait 2 seconds each time
)
```

### Using LLMClient

```python
from llm import LLMClient

client = LLMClient(
    openai_api_key="...",
    ark_api_key="...",
)

content = client.completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## 📊 Return Value Rules

| Scenario | return_full_response | Return Type | Description |
|----------|---------------------|-------------|-------------|
| Normal model | False (default) | `str` | Directly returns text content |
| Has thinking content | False (default) | `dict` | `{"content": "...", "thinking_content": "..."}` |
| Any model | True | `Response object` | Full API response |

## 🧠 Reasoning_Effort Parameter Comparison

| Provider | none/minimal | low | medium | high | Special Notes |
|----------|-------------|-----|--------|------|---------------|
| OpenAI | No parameter | ✓ | ✓ | ✓ | gpt-4.1 not supported |
| Ark | minimal | low | medium | high | Auto conversion |
| DashScope | Disabled/disabled | ✓ | ✓ | ✓ | Use streaming for thinking |
| Gemini 3 | low (cannot disable) | ✓ | →low | ✓ | Does not support medium |
| Gemini 2.5 Pro | 128 (minimum) | 2048 | 8192 | 32768 | Cannot completely disable |
| Gemini 2.5 Flash | 0 (can disable) | 4096 | 12288 | 24576 | Can completely disable |
| DeepSeek | Invalid | Invalid | Invalid | Invalid | Model decides automatically |

## 🎯 Best Practices

### 1. Simple Scenario
```python
# Only need text content
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 2. Handle Thinking Content
```python
result = completion(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Complex question"}]
)

if isinstance(result, dict):
    print(f"Answer: {result['content']}")
    print(f"Thinking: {result['thinking_content']}")
else:
    print(f"Answer: {result}")
```

### 3. Need Metadata
```python
response = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Hello"}],
    return_full_response=True
)
print(f"Usage: {response.usage.total_tokens} tokens")
```

## 📖 Detailed Documentation

- [Usage Guide](./USAGE_GUIDE.md) - Detailed usage instructions
- [OpenAI Example](./openai_example.py)
- [Ark Example](./ark_example.py)
- [DashScope Example](./dashscope_example.py)
- [Gemini Example](./gemini_example.py)
- [DeepSeek Example](./deepseek_example.py)

## 🔧 Technical Architecture

### Lazy Loading Mechanism
- Providers are initialized only on first use
- Saves resources, improves startup speed

### Auto Mapping
- Model names automatically map to corresponding providers
- Adding new providers doesn't require modifying routing logic

### Error Retry
- Built-in retry mechanism, default 3 retries
- Configurable retry times and delay

### Parameter Conversion
- Automatically converts unified parameters to different provider formats
- Friendly warning messages

## 🎨 Multimodal Input Support

Support for image and video input across providers:

### Image Input

| Provider | Support | Description |
|---------|---------|-------------|
| **OpenAI** | ✅ Supported | Supports URL, Base64, local path (auto converts to Base64) |
| **Gemini** | ✅ Supported | Supports URL, Base64, local path (auto handles) |
| **Ark** | ✅ Supported | Supports URL, Base64, local path (auto converts to Base64) |
| **DashScope** | ✅ Supported | Supports URL, Base64, local path (auto converts to Base64) |
| **DeepSeek** | ❌ Not supported | Warns and skips images |

### Video Input

| Provider | Support | Processing Method |
|---------|---------|-------------------|
| **OpenAI** | ✅ Supported | Auto frame extraction to multiple images (supports fps, max_frames parameters) |
| **Gemini** | ✅ Supported | Native support, <20MB uses inline, ≥20MB auto uploads |
| **Ark** | ✅ Supported | Converts to Base64, 50MB limit (auto compresses if exceeded) |
| **DashScope** | ✅ Supported | Converts to Base64, 10MB limit (auto compresses if exceeded) |
| **DeepSeek** | ❌ Not supported | Warns and skips videos |

### Usage Examples

#### Image Input

```python
from llm import completion

# All supported providers can use this way
result = completion(
    model="gpt-4o",  # Or other supported models
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "/path/to/image.jpg"  # Local path, URL or Base64
                }
            },
            {
                "type": "text",
                "text": "Describe this image"
            }
        ]
    }]
)
```

#### Video Input (OpenAI)

```python
from llm import completion

# OpenAI supports video through frame extraction
result = completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video": "/path/to/video.mp4",
                "fps": 2.0,         # Optional: frames per second
                "max_frames": 40,   # Optional: maximum frames
            },
            {
                "type": "text",
                "text": "Describe video content"
            }
        ]
    }]
)
```

#### Video Input (Gemini)

```python
from llm import completion

# Gemini natively supports video
result = completion(
    model="gemini-2.5-flash",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # Auto handles large files
                }
            },
            {
                "type": "text",
                "text": "Analyze video content"
            }
        ]
    }]
)
```

#### Video Input (Ark/DashScope)

```python
from llm import completion

# Ark supports video (50MB limit, auto compresses if exceeded)
result = completion(
    model="doubao-seed-1-6-vision-250815",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # Local path or URL, auto converts to Base64
                }
            },
            {
                "type": "text",
                "text": "Describe video content"
            }
        ]
    }]
)

# DashScope supports video (10MB limit, auto compresses if exceeded)
result = completion(
    model="qwen3-vl-8b-instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # Local path or URL, auto converts to Base64
                }
            },
            {
                "type": "text",
                "text": "Summarize video"
            }
        ]
    }]
)
```

**Notes**:
- Ark video size limit 50MB, auto compresses and warns if exceeded
- DashScope video size limit 10MB, auto compresses and warns if exceeded
- Base64 format videos are passed directly without processing
- Local paths and URLs are automatically converted to Base64 format

## 📝 Adding New Provider

1. Inherit `BaseLLMProvider`
2. Implement `completion()` method
3. Define `SUPPORTED_MODELS` list
4. Register in `client.py`

See the implementation code of each provider for details.

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT
