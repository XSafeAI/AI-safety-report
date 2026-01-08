# GeminiEvaluation

一个用于评估多种大语言模型在多模态安全基准测试上表现的完整评测系统。

## ✨ 项目特点

- 🤖 **多模型支持**: 支持 Gemini、GPT、DeepSeek、DashScope/Qwen、Ark/豆包、XAI/Grok 等主流 LLM
- 📊 **多数据集**: 集成 8+ 个主流多模态安全测试数据集
- 🔄 **统一接口**: 基于统一的 LLM 调用库，一致的 API 体验
- 🎯 **自动评测**: 使用 Qwen3Guard-Gen-8B 作为评判模型，自动计算安全性指标
- ⚡ **高效并发**: 支持异步并发评测，提升评测效率
- 🛠️ **完整流程**: 从数据预处理到结果分析的完整工作流
- 🔁 **错误重试**: 内置重试机制，提高评测稳定性

## 📦 支持的数据集

| 数据集 | 说明 | 适配器 |
|--------|------|--------|
| **VLJailbreakBench** | 视觉语言越狱测试基准 | `vljailbreakbench_adapter.py` |
| **USB** | 通用安全基准 | `usb_adapter.py` |
| **MIS_Test** | 多模态不安全性测试 | `mis_adapter.py` |
| **MM-SafetyBench** | 多模态安全基准 | `mm_safetybench_adapter.py` |
| **MemeSafetyBench** | 表情包安全基准 | `memesafetybench_adapter.py` |
| **SIUO** | 安全性交互理解 | `siuo_adapter.py` |
| **JailbreakV-28k** | 越狱测试数据集 | `jailbreakv_adapter.py` |

## 🚀 快速开始

### 1. 环境配置

**系统要求**：
- Python 3.8+
- CUDA 11.8+ (用于本地评判模型)
- ffmpeg (用于视频处理，可选)

**安装依赖**：

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 ffmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### 2. 配置 API Keys

复制环境变量模板并配置你的 API Keys：

```bash
# 复制模板文件
cp env_template.txt .env

# 编辑 .env 文件，填入你的 API Keys
vim .env  # 或使用其他编辑器
```

模板文件 `env_template.txt` 包含了所有支持的 API 配置项，包括：
- Gemini API (Google AI)
- OpenAI API
- DeepSeek API
- DashScope API (阿里云通义千问)
- Ark API (火山引擎豆包)
- XAI API (Grok)
- SiliconFlow API
- 代理设置、日志级别等可选配置

### 3. 数据集准备

**下载数据集**：

```bash
# 使用提供的下载脚本
bash script/download.sh
```

**处理数据集**：

```bash
# 处理所有数据集
bash script/process_data.sh

# 或处理单个数据集
python evaluation/process_datasets.py --dataset vljailbreakbench
```

处理后的数据将保存在 `workspace/data/processed/` 目录下。

### 4. 运行评测

**方式一：使用脚本批量评测**

```bash
# 评测单个模型的所有数据集
bash script/evaluate.sh --model gemini-3-pro-preview

# 指定数据集
bash script/evaluate.sh --model gpt-5-mini --datasets vljailbreakbench,usb

# 自定义参数
bash script/evaluate.sh \
  --model deepseek-reasoner \
  --datasets usb \
  --max-samples 1000 \
  --concurrency 10 \
  --reasoning-effort high \
  --max-tokens 512
```

**方式二：Python 脚本单独评测**

```bash
# 基础评测
python evaluation/evaluate.py \
  --model gemini-3-pro-preview \
  --dataset vljailbreakbench \
  --max-samples 100 \
  --concurrency 5

# 重试失败样本
python evaluation/evaluate.py \
  --retry-errors workspace/results/gemini-3-pro-preview/vljailbreakbench_20260108_120000.jsonl
```

### 5. 计算指标

评测完成后，使用本地评判模型计算安全性指标：

```bash
# 计算单个结果文件的指标
python evaluation/compute_metrics.py \
  --result-file workspace/results/gemini-3-pro-preview/vljailbreakbench_20260108_120000.jsonl \
  --dataset vljailbreakbench

# 批量计算所有结果
bash script/compute_all_metrics.sh --model gemini-3-pro-preview
```

### 6. 生成报告

```bash
# 生成评测报告
python evaluation/generate_report.py \
  --model gemini-3-pro-preview \
  --output workspace/results/gemini-3-pro-preview/report.html
```

## 📖 项目结构

```
GeminiEvaluation/
├── evaluation/              # 评测核心代码
│   ├── adapters/           # 数据集适配器
│   │   ├── base_adapter.py          # 基础适配器
│   │   ├── vljailbreakbench_adapter.py
│   │   ├── usb_adapter.py
│   │   └── ...
│   ├── evaluate.py         # 评测主脚本
│   ├── evaluate_thread.py  # 多线程评测
│   ├── dataset_loader.py   # 数据加载器
│   ├── process_datasets.py # 数据预处理
│   ├── compute_metrics.py  # 指标计算
│   └── generate_report.py  # 报告生成
│
├── llm/                    # 统一 LLM 调用库
│   ├── client.py          # 统一客户端
│   ├── base.py            # 基础类
│   ├── gemini_provider.py  # Gemini 提供商
│   ├── openai_provider.py  # OpenAI 提供商
│   ├── deepseek_provider.py # DeepSeek 提供商
│   ├── dashscope_provider.py # DashScope 提供商
│   ├── ark_provider.py     # Ark 提供商
│   ├── xai_provider.py     # XAI 提供商
│   └── README.md          # LLM 库详细文档
│
├── script/                 # 实用脚本
│   ├── download.sh        # 数据下载
│   ├── process_data.sh    # 数据处理
│   ├── evaluate.sh        # 批量评测
│   ├── evaluate_thread.sh # 多线程评测
│   ├── compute_all_metrics.sh # 批量计算指标
│   └── retry_errors_example.sh # 重试示例
│
├── workspace/             # 工作目录
│   ├── data/             # 数据目录
│   │   ├── raw/         # 原始数据
│   │   ├── processed/   # 处理后数据
│   │   └── temp/        # 临时文件
│   └── results/         # 评测结果
│       ├── gemini-3-pro-preview/
│       ├── gpt-5-mini/
│       └── metrics_summary.xlsx
│
├── external/              # 外部模型
│   └── model/
│       ├── Qwen3Guard-Gen-8B/      # 评判模型
│       └── Qwen2.5-VL-7B-Instruct/ # 备用模型
│
├── requirements.txt       # Python 依赖
├── env_template.txt       # 环境变量配置模板
├── .env                   # 环境变量配置 (需自行创建)
└── README.md             # 项目文档
```

## 🎯 支持的模型

### Gemini 系列
- `gemini-3-pro-preview` - Gemini 3 Pro (预览版)
- `gemini-2.5-pro` - Gemini 2.5 Pro
- `gemini-2.5-flash` - Gemini 2.5 Flash

### OpenAI 系列
- `gpt-5` - GPT-5
- `gpt-5-mini` - GPT-5 Mini
- `gpt-4.1` - GPT-4.1
- `gpt-4o` - GPT-4 Optimized
- `gpt-4o-mini` - GPT-4o Mini

### DeepSeek 系列
- `deepseek-reasoner` - DeepSeek Reasoner (带思考链)
- `deepseek-chat` - DeepSeek Chat

### DashScope/通义千问系列
- `qwen3-vl-*-thinking` - Qwen3 VL 思考系列
- `qwen3-vl-*-instruct` - Qwen3 VL 指令系列
- `qwen2.5-vl-*-instruct` - Qwen2.5 VL 系列

### Ark/火山引擎豆包系列
- `doubao-seed-1-6-251015` - 豆包 Seed 文本模型
- `doubao-seed-1-6-vision-250815` - 豆包 Seed 视觉模型

### XAI/Grok 系列
- `grok-4-1-fast-non-reasoning` - Grok 4.1 快速模式
- `grok-4-1-reasoning` - Grok 4.1 推理模式

更多模型支持请查看 [llm/README.md](llm/README.md)

## 📊 评测指标

本项目使用 Qwen3Guard-Gen-8B 作为评判模型，计算以下指标：

- **Attack Success Rate (ASR)**: 攻击成功率 = Unsafe 回答数 / 总样本数
- **Refusal Rate**: 拒绝率 = 拒绝回答数 / 总样本数
- **Safety Rate**: 安全率 = Safe 回答数 / 总样本数
- **Error Rate**: 错误率 = 评测失败数 / 总样本数

## 🔧 高级用法

### 自定义数据集适配器

创建新的数据集适配器，继承 `BaseDatasetAdapter`：

```python
from evaluation.adapters.base_adapter import BaseDatasetAdapter

class MyDatasetAdapter(BaseDatasetAdapter):
    """自定义数据集适配器"""
    
    def load_dataset(self) -> List[Dict]:
        """加载数据集"""
        # 实现数据加载逻辑
        pass
    
    def format_sample(self, sample: Dict) -> Dict:
        """格式化样本"""
        return {
            'prompt': sample['question'],
            'images': [sample['image_path']],
            'meta': {
                'original_id': sample['id'],
                'category': sample.get('category', 'unknown')
            }
        }
```

### 并发控制

```bash
# 串行执行 (适合调试)
python evaluation/evaluate.py --concurrency 1

# 低并发 (适合 API 限流严格的服务)
python evaluation/evaluate.py --concurrency 5

# 高并发 (适合不限流或本地部署的模型)
python evaluation/evaluate.py --concurrency 32
```

### 推理强度控制

```bash
# 低强度 (快速但可能不够准确)
python evaluation/evaluate.py --reasoning-effort low

# 中等强度 (平衡速度和准确性)
python evaluation/evaluate.py --reasoning-effort medium

# 高强度 (慢但更准确)
python evaluation/evaluate.py --reasoning-effort high
```

### 错误重试机制

```bash
# 自动重试失败的样本
python evaluation/evaluate.py \
  --retry-errors workspace/results/model/dataset.jsonl \
  --retry-times 3 \
  --retry-delay 5.0
```

## 📝 数据格式

### 处理后数据格式 (JSONL)

```json
{
  "prompt": "问题文本",
  "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "meta": {
    "original_id": "sample_001",
    "dataset": "vljailbreakbench",
    "category": "unsafe_content",
    "task_type": "jailbreak"
  }
}
```

### 评测结果格式 (JSONL)

```json
{
  "prompt": "问题文本",
  "images": ["/path/to/image1.jpg"],
  "response": "模型回答",
  "meta": {
    "original_id": "sample_001",
    "dataset": "vljailbreakbench",
    "category": "unsafe_content"
  }
}
```

### 指标结果格式 (JSONL)

```json
{
  "prompt": "问题文本",
  "response": "模型回答",
  "safe_label": "Unsafe",
  "refusal_label": "No",
  "judge_output": "Safety: Unsafe\nRefusal: No\n...",
  "meta": {...}
}
```

## 🐛 常见问题

### Q: API 调用频繁失败怎么办？

A: 可以：
1. 降低并发数 `--concurrency 1`
2. 增加重试次数和延迟 `--retry-times 5 --retry-delay 10.0`
3. 使用错误重试功能重新评测失败样本

### Q: 如何验证图片路径是否正确？

A: 运行验证脚本：

```bash
python evaluation/verify_image_paths.py --dataset vljailbreakbench
```

### Q: 如何添加新的 LLM 提供商？

A: 参考 [llm/README.md](llm/README.md) 中的"添加新 Provider"章节。

### Q: 评测结果保存在哪里？

A: 结果保存在 `workspace/results/<model_name>/` 目录下，文件名格式为 `<dataset>_<timestamp>.jsonl`。

### Q: 如何查看某个模型的所有评测结果？

A: 查看对应的模型目录：

```bash
ls -lh workspace/results/gemini-3-pro-preview/
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**: 本项目仅用于学术研究和安全评测，请勿用于任何非法或不当目的。
