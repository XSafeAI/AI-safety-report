# LLM统一调用库

一个支持多种大语言模型的统一调用库，提供简洁一致的API接口。

## ✨ 核心特性

- 🚀 **统一接口**: 所有模型使用相同的`completion()`方法
- 🔄 **自动路由**: 根据模型名称自动选择对应的provider
- 🧠 **思考支持**: 统一的思考参数接口，自动处理不同模型的差异
- 🔁 **自动重试**: 内置重试机制，提高稳定性
- 💡 **懒加载**: Provider按需初始化，节省资源
- 🎯 **简洁返回**: 默认返回解析后的文本，可选返回完整response

## 📦 支持的模型

### 1. OpenAI (OpenAIProvider)
- **模型**: gpt-5, gpt-5-mini, gpt-4.1, gpt-4o, gpt-4o-mini
- **思考支持**: gpt-5系列支持reasoning_effort (low/medium/high)
- **环境变量**: `OPENAI_API_KEY`

### 2. Ark/火山引擎豆包 (ArkProvider)
- **模型**: doubao-seed-1-6-251015, doubao-seed-1-6-vision-250815
- **思考支持**: 
  - doubao-seed-1-6-251015: 支持reasoning_effort分级
  - doubao-seed-1-6-vision-250815: 支持extra_body思考开关
- **环境变量**: `ARK_API_KEY`
- **特点**: 返回思考内容

### 3. DashScope/通义千问 (DashScopeProvider)
- **模型**: 
  - 只支持思考: qwen3-vl-*-thinking系列
  - 不支持思考: qwen3-vl-*-instruct系列
  - 灵活模型: qwen2.5-vl/qwen2.5系列
- **思考支持**: 
  - 思考模型自动使用流式API
  - 灵活模型可通过reasoning_effort控制
- **环境变量**: `DASHSCOPE_API_KEY`
- **特点**: 自动处理流式响应

### 4. Gemini (GeminiProvider)
- **模型**: 
  - Gemini 3: gemini-3-pro-preview (使用thinkingLevel)
  - Gemini 2.5: gemini-2.5-pro, gemini-2.5-flash (使用thinkingBudget)
- **思考支持**: 
  - Gemini 3: low/high级别
  - Gemini 2.5 Pro: 无法完全停用 (最小值128)
  - Gemini 2.5 Flash: 可停用 (设为0)
- **环境变量**: `GEMINI_API_KEY`
- **特点**: 返回思考总结

### 5. DeepSeek (DeepSeekProvider)
- **模型**: 
  - deepseek-reasoner: 支持思考
  - deepseek-chat: 不支持思考
- **思考支持**: 
  - deepseek-reasoner自动思考，无法控制
  - reasoning_effort参数无效
- **多模态支持**: ❌ 不支持图片和视频输入
- **环境变量**: `DEEPSEEK_API_KEY`
- **特点**: 思考行为由模型内置，仅支持文本输入

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

**系统依赖**：
- 如果需要使用视频功能（压缩、抽帧等），需要安装 ffmpeg：
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  
  # Windows
  # 从 https://ffmpeg.org/download.html 下载并安装
  ```

### 设置环境变量

```bash
export OPENAI_API_KEY="your-key"
export ARK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
```

### 基础使用

```python
from llm import completion

# 默认返回文本（推荐）
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}]
)
print(content)  # 直接打印文本
```

### 使用思考功能

```python
# OpenAI
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "什么是量子计算？"}],
    reasoning_effort="high"
)

# Ark（返回字典）
result = completion(
    model="doubao-seed-1-6-251015",
    messages=[{"role": "user", "content": "复杂问题"}],
    reasoning_effort="medium"
)
print(result["content"])           # 答案
print(result["thinking_content"])  # 思考过程

# DeepSeek（自动思考）
result = completion(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "9.11和9.8哪个大？"}]
)
print(result["content"])           # 答案
print(result["thinking_content"])  # 思考过程
```

### 返回完整Response

```python
response = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}],
    return_full_response=True
)
print(response.usage)   # token用量
print(response.model)   # 模型信息
```

### 自动重试

```python
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}],
    retry_times=3,      # 重试3次
    retry_delay=2.0     # 每次等待2秒
)
```

### 使用LLMClient

```python
from llm import LLMClient

client = LLMClient(
    openai_api_key="...",
    ark_api_key="...",
)

content = client.completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}]
)
```

## 📊 返回值规则

| 场景 | return_full_response | 返回值类型 | 说明 |
|------|---------------------|-----------|------|
| 普通模型 | False (默认) | `str` | 直接返回文本内容 |
| 有思考内容 | False (默认) | `dict` | `{"content": "...", "thinking_content": "..."}` |
| 任意模型 | True | `Response对象` | 完整的API响应 |

## 🧠 Reasoning_Effort参数对比

| Provider | none/minimal | low | medium | high | 特殊说明 |
|----------|-------------|-----|--------|------|---------|
| OpenAI | 不传参数 | ✓ | ✓ | ✓ | gpt-4.1不支持 |
| Ark | minimal | low | medium | high | 自动转换 |
| DashScope | 停用/disabled | ✓ | ✓ | ✓ | 思考时用流式 |
| Gemini 3 | low (无法停用) | ✓ | →low | ✓ | 不支持medium |
| Gemini 2.5 Pro | 128 (最小值) | 2048 | 8192 | 32768 | 无法完全停用 |
| Gemini 2.5 Flash | 0 (可停用) | 4096 | 12288 | 24576 | 可完全停用 |
| DeepSeek | 无效 | 无效 | 无效 | 无效 | 模型自动决定 |

## 🎯 最佳实践

### 1. 简单场景
```python
# 只需要文本内容
content = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 2. 处理思考内容
```python
result = completion(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "复杂问题"}]
)

if isinstance(result, dict):
    print(f"答案: {result['content']}")
    print(f"思考: {result['thinking_content']}")
else:
    print(f"答案: {result}")
```

### 3. 需要元数据
```python
response = completion(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "你好"}],
    return_full_response=True
)
print(f"用量: {response.usage.total_tokens} tokens")
```

## 📖 详细文档

- [使用指南](./USAGE_GUIDE.md) - 详细的使用说明
- [OpenAI示例](./openai_example.py)
- [Ark示例](./ark_example.py)
- [DashScope示例](./dashscope_example.py)
- [Gemini示例](./gemini_example.py)
- [DeepSeek示例](./deepseek_example.py)

## 🔧 技术架构

### 懒加载机制
- Provider只在第一次使用时初始化
- 节省资源，提高启动速度

### 自动映射
- 模型名称自动映射到对应的provider
- 添加新provider无需修改路由逻辑

### 错误重试
- 内置重试机制，默认重试3次
- 可配置重试次数和延迟时间

### 参数转换
- 自动转换统一参数到不同provider的格式
- 友好的警告提示

## 🎨 多模态输入支持

各Provider对图片和视频输入的支持情况：

### 图片输入

| Provider | 支持情况 | 说明 |
|---------|---------|------|
| **OpenAI** | ✅ 支持 | 支持URL、Base64、本地路径（自动转Base64） |
| **Gemini** | ✅ 支持 | 支持URL、Base64、本地路径（自动处理） |
| **Ark** | ✅ 支持 | 支持URL、Base64、本地路径（自动转Base64） |
| **DashScope** | ✅ 支持 | 支持URL、Base64、本地路径（自动转Base64） |
| **DeepSeek** | ❌ 不支持 | 遇到图片会警告并跳过 |

### 视频输入

| Provider | 支持情况 | 处理方式 |
|---------|---------|----------|
| **OpenAI** | ✅ 支持 | 自动抽帧转多图输入（支持fps、max_frames参数） |
| **Gemini** | ✅ 支持 | 原生支持，<20MB用inline，≥20MB自动上传 |
| **Ark** | ✅ 支持 | 转Base64，限制50MB（超过自动压缩） |
| **DashScope** | ✅ 支持 | 转Base64，限制10MB（超过自动压缩） |
| **DeepSeek** | ❌ 不支持 | 遇到视频会警告并跳过 |

### 使用示例

#### 图片输入

```python
from llm import completion

# 所有支持的Provider都可以这样使用
result = completion(
    model="gpt-4o",  # 或其他支持的模型
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "/path/to/image.jpg"  # 本地路径、URL或Base64
                }
            },
            {
                "type": "text",
                "text": "描述这张图片"
            }
        ]
    }]
)
```

#### 视频输入（OpenAI）

```python
from llm import completion

# OpenAI通过抽帧方式支持视频
result = completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video": "/path/to/video.mp4",
                "fps": 2.0,         # 可选：每秒抽取帧数
                "max_frames": 40,   # 可选：最大帧数
            },
            {
                "type": "text",
                "text": "描述视频内容"
            }
        ]
    }]
)
```

#### 视频输入（Gemini）

```python
from llm import completion

# Gemini原生支持视频
result = completion(
    model="gemini-2.5-flash",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # 自动处理大小文件
                }
            },
            {
                "type": "text",
                "text": "分析视频内容"
            }
        ]
    }]
)
```

#### 视频输入（Ark/DashScope）

```python
from llm import completion

# Ark支持视频（限制50MB，超过自动压缩）
result = completion(
    model="doubao-seed-1-6-vision-250815",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # 本地路径或URL，自动转Base64
                }
            },
            {
                "type": "text",
                "text": "描述视频内容"
            }
        ]
    }]
)

# DashScope支持视频（限制10MB，超过自动压缩）
result = completion(
    model="qwen3-vl-8b-instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "/path/to/video.mp4"  # 本地路径或URL，自动转Base64
                }
            },
            {
                "type": "text",
                "text": "总结视频"
            }
        ]
    }]
)
```

**注意事项**：
- Ark 视频大小限制 50MB，超过会自动压缩并发出警告
- DashScope 视频大小限制 10MB，超过会自动压缩并发出警告
- Base64格式的视频不做处理，直接传递
- 本地路径和URL会自动转换为Base64格式

## 📝 添加新Provider

1. 继承`BaseLLMProvider`
2. 实现`completion()`方法
3. 定义`SUPPORTED_MODELS`列表
4. 在`client.py`中注册

详见各provider的实现代码。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 License

MIT

