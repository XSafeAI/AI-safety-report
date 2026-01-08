# GeminiEvaluation

A comprehensive evaluation system for assessing the performance of multiple Large Language Models on multimodal safety benchmarks.

## ✨ Key Features

- 🤖 **Multi-Model Support**: Supports mainstream LLMs including Gemini, GPT, DeepSeek, DashScope/Qwen, Ark/Doubao, XAI/Grok
- 📊 **Multiple Datasets**: Integrates 8+ mainstream multimodal safety testing datasets
- 🔄 **Unified Interface**: Based on a unified LLM calling library with consistent API experience
- 🎯 **Automatic Evaluation**: Uses Qwen3Guard-Gen-8B as the judge model to automatically calculate safety metrics
- ⚡ **Efficient Concurrency**: Supports asynchronous concurrent evaluation to improve efficiency
- 🛠️ **Complete Workflow**: Complete pipeline from data preprocessing to result analysis
- 🔁 **Error Retry**: Built-in retry mechanism to improve evaluation stability

## 📦 Supported Datasets

| Dataset | Description | Adapter |
|---------|-------------|---------|
| **VLJailbreakBench** | Vision-Language Jailbreak Testing Benchmark | `vljailbreakbench_adapter.py` |
| **USB** | Universal Safety Benchmark | `usb_adapter.py` |
| **MIS_Test** | Multimodal Unsafety Test | `mis_adapter.py` |
| **MM-SafetyBench** | Multimodal Safety Benchmark | `mm_safetybench_adapter.py` |
| **MemeSafetyBench** | Meme Safety Benchmark | `memesafetybench_adapter.py` |
| **SIUO** | Safety Interaction Understanding | `siuo_adapter.py` |
| **JailbreakV-28k** | Jailbreak Test Dataset | `jailbreakv_adapter.py` |

## 🚀 Quick Start

### 1. Environment Setup

**System Requirements**:
- Python 3.8+
- CUDA 11.8+ (for local judge model)
- ffmpeg (for video processing, optional)

**Install Dependencies**:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### 2. Configure API Keys

Copy the environment variable template and configure your API Keys:

```bash
# Copy template file
cp env_template.txt .env

# Edit .env file and fill in your API Keys
vim .env  # or use other editors
```

The template file `env_template.txt` contains all supported API configuration items, including:
- Gemini API (Google AI)
- OpenAI API
- DeepSeek API
- DashScope API (Alibaba Cloud Qwen)
- Ark API (Volcano Engine Doubao)
- XAI API (Grok)
- SiliconFlow API
- Optional configurations like proxy settings, log level, etc.

### 3. Dataset Preparation

**Download Datasets**:

```bash
# Use the provided download script
bash script/download.sh
```

**Process Datasets**:

```bash
# Process all datasets
bash script/process_data.sh

# Or process a single dataset
python evaluation/process_datasets.py --dataset vljailbreakbench
```

Processed data will be saved in the `workspace/data/processed/` directory.

### 4. Run Evaluation

**Method 1: Batch Evaluation Using Scripts**

```bash
# Evaluate all datasets for a single model
bash script/evaluate.sh --model gemini-3-pro-preview

# Specify datasets
bash script/evaluate.sh --model gpt-5-mini --datasets vljailbreakbench,usb

# Custom parameters
bash script/evaluate.sh \
  --model deepseek-reasoner \
  --datasets usb \
  --max-samples 1000 \
  --concurrency 10 \
  --reasoning-effort high \
  --max-tokens 512
```

**Method 2: Individual Evaluation Using Python Script**

```bash
# Basic evaluation
python evaluation/evaluate.py \
  --model gemini-3-pro-preview \
  --dataset vljailbreakbench \
  --max-samples 100 \
  --concurrency 5

# Retry failed samples
python evaluation/evaluate.py \
  --retry-errors workspace/results/gemini-3-pro-preview/vljailbreakbench_20260108_120000.jsonl
```

### 5. Calculate Metrics

After evaluation is complete, use the local judge model to calculate safety metrics:

```bash
# Calculate metrics for a single result file
python evaluation/compute_metrics.py \
  --result-file workspace/results/gemini-3-pro-preview/vljailbreakbench_20260108_120000.jsonl \
  --dataset vljailbreakbench

# Batch calculate all results
bash script/compute_all_metrics.sh --model gemini-3-pro-preview
```

### 6. Generate Report

```bash
# Generate evaluation report
python evaluation/generate_report.py \
  --model gemini-3-pro-preview \
  --output workspace/results/gemini-3-pro-preview/report.html
```

## 📖 Project Structure

```
GeminiEvaluation/
├── evaluation/              # Core evaluation code
│   ├── adapters/           # Dataset adapters
│   │   ├── base_adapter.py          # Base adapter
│   │   ├── vljailbreakbench_adapter.py
│   │   ├── usb_adapter.py
│   │   └── ...
│   ├── evaluate.py         # Main evaluation script
│   ├── evaluate_thread.py  # Multi-threaded evaluation
│   ├── dataset_loader.py   # Data loader
│   ├── process_datasets.py # Data preprocessing
│   ├── compute_metrics.py  # Metrics calculation
│   └── generate_report.py  # Report generation
│
├── llm/                    # Unified LLM calling library
│   ├── client.py          # Unified client
│   ├── base.py            # Base classes
│   ├── gemini_provider.py  # Gemini provider
│   ├── openai_provider.py  # OpenAI provider
│   ├── deepseek_provider.py # DeepSeek provider
│   ├── dashscope_provider.py # DashScope provider
│   ├── ark_provider.py     # Ark provider
│   ├── xai_provider.py     # XAI provider
│   └── README.md          # Detailed LLM library documentation
│
├── script/                 # Utility scripts
│   ├── download.sh        # Data download
│   ├── process_data.sh    # Data processing
│   ├── evaluate.sh        # Batch evaluation
│   ├── evaluate_thread.sh # Multi-threaded evaluation
│   ├── compute_all_metrics.sh # Batch metrics calculation
│   └── retry_errors_example.sh # Retry example
│
├── workspace/             # Working directory
│   ├── data/             # Data directory
│   │   ├── raw/         # Raw data
│   │   ├── processed/   # Processed data
│   │   └── temp/        # Temporary files
│   └── results/         # Evaluation results
│       ├── gemini-3-pro-preview/
│       ├── gpt-5-mini/
│       └── metrics_summary.xlsx
│
├── external/              # External models
│   └── model/
│       ├── Qwen3Guard-Gen-8B/      # Judge model
│       └── Qwen2.5-VL-7B-Instruct/ # Backup model
│
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variable configuration template
├── .env                   # Environment variable configuration (need to create manually)
└── README.md             # Project documentation
```

## 🎯 Supported Models

### Gemini Series
- `gemini-3-pro-preview` - Gemini 3 Pro (Preview)
- `gemini-2.5-pro` - Gemini 2.5 Pro
- `gemini-2.5-flash` - Gemini 2.5 Flash

### OpenAI Series
- `gpt-5` - GPT-5
- `gpt-5-mini` - GPT-5 Mini
- `gpt-4.1` - GPT-4.1
- `gpt-4o` - GPT-4 Optimized
- `gpt-4o-mini` - GPT-4o Mini

### DeepSeek Series
- `deepseek-reasoner` - DeepSeek Reasoner (with reasoning chain)
- `deepseek-chat` - DeepSeek Chat

### DashScope/Qwen Series
- `qwen3-vl-*-thinking` - Qwen3 VL Thinking Series
- `qwen3-vl-*-instruct` - Qwen3 VL Instruct Series
- `qwen2.5-vl-*-instruct` - Qwen2.5 VL Series

### Ark/Volcano Engine Doubao Series
- `doubao-seed-1-6-251015` - Doubao Seed Text Model
- `doubao-seed-1-6-vision-250815` - Doubao Seed Vision Model

### XAI/Grok Series
- `grok-4-1-fast-non-reasoning` - Grok 4.1 Fast Mode
- `grok-4-1-reasoning` - Grok 4.1 Reasoning Mode

For more model support, please see [llm/README.md](llm/README.md)

## 📊 Evaluation Metrics

This project uses Qwen3Guard-Gen-8B as the judge model to calculate the following metrics:

- **Attack Success Rate (ASR)**: Attack success rate = Number of unsafe responses / Total samples
- **Refusal Rate**: Refusal rate = Number of refusals / Total samples
- **Safety Rate**: Safety rate = Number of safe responses / Total samples
- **Error Rate**: Error rate = Number of evaluation failures / Total samples

## 🔧 Advanced Usage

### Custom Dataset Adapter

Create a new dataset adapter by inheriting `BaseDatasetAdapter`:

```python
from evaluation.adapters.base_adapter import BaseDatasetAdapter

class MyDatasetAdapter(BaseDatasetAdapter):
    """Custom dataset adapter"""
    
    def load_dataset(self) -> List[Dict]:
        """Load dataset"""
        # Implement data loading logic
        pass
    
    def format_sample(self, sample: Dict) -> Dict:
        """Format sample"""
        return {
            'prompt': sample['question'],
            'images': [sample['image_path']],
            'meta': {
                'original_id': sample['id'],
                'category': sample.get('category', 'unknown')
            }
        }
```

### Concurrency Control

```bash
# Serial execution (suitable for debugging)
python evaluation/evaluate.py --concurrency 1

# Low concurrency (suitable for services with strict API rate limits)
python evaluation/evaluate.py --concurrency 5

# High concurrency (suitable for unlimited or locally deployed models)
python evaluation/evaluate.py --concurrency 32
```

### Reasoning Effort Control

```bash
# Low effort (fast but may be less accurate)
python evaluation/evaluate.py --reasoning-effort low

# Medium effort (balance between speed and accuracy)
python evaluation/evaluate.py --reasoning-effort medium

# High effort (slow but more accurate)
python evaluation/evaluate.py --reasoning-effort high
```

### Error Retry Mechanism

```bash
# Automatically retry failed samples
python evaluation/evaluate.py \
  --retry-errors workspace/results/model/dataset.jsonl \
  --retry-times 3 \
  --retry-delay 5.0
```

## 📝 Data Format

### Processed Data Format (JSONL)

```json
{
  "prompt": "Question text",
  "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
  "meta": {
    "original_id": "sample_001",
    "dataset": "vljailbreakbench",
    "category": "unsafe_content",
    "task_type": "jailbreak"
  }
}
```

### Evaluation Result Format (JSONL)

```json
{
  "prompt": "Question text",
  "images": ["/path/to/image1.jpg"],
  "response": "Model response",
  "meta": {
    "original_id": "sample_001",
    "dataset": "vljailbreakbench",
    "category": "unsafe_content"
  }
}
```

### Metrics Result Format (JSONL)

```json
{
  "prompt": "Question text",
  "response": "Model response",
  "safe_label": "Unsafe",
  "refusal_label": "No",
  "judge_output": "Safety: Unsafe\nRefusal: No\n...",
  "meta": {...}
}
```

## 🐛 Common Issues

### Q: What to do if API calls frequently fail?

A: You can:
1. Reduce concurrency `--concurrency 1`
2. Increase retry times and delay `--retry-times 5 --retry-delay 10.0`
3. Use error retry feature to re-evaluate failed samples

### Q: How to verify image paths are correct?

A: Run the verification script:

```bash
python evaluation/verify_image_paths.py --dataset vljailbreakbench
```

### Q: How to add a new LLM provider?

A: Refer to the "Adding New Provider" section in [llm/README.md](llm/README.md).

### Q: Where are evaluation results saved?

A: Results are saved in the `workspace/results/<model_name>/` directory with filenames formatted as `<dataset>_<timestamp>.jsonl`.

### Q: How to view all evaluation results for a model?

A: View the corresponding model directory:

```bash
ls -lh workspace/results/gemini-3-pro-preview/
```

## 📄 License

MIT License

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📮 Contact

If you have any questions or suggestions, please contact us through Issues.

---

**Note**: This project is for academic research and safety evaluation only. Do not use it for any illegal or inappropriate purposes.
