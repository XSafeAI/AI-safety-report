# T2I Regulatory Compliance Evaluation

A toolkit for assessing safety compliance of Text-to-Image models (e.g., Gemini-3, Seedream) using Qwen3-VL as an automated judge.

## 📦 Setup

**Dependencies**
```bash
pip install openai google-genai pyyaml
```

**Configuration**
Configure your API keys in `config/config.yaml`:

```yaml
gemini:
  model_name: "gemini-3-pro-image-preview"
  api_key: "YOUR_KEY"

seedream:
  model_name: "doubao-seedream-4-5-251128"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  api_key: "YOUR_KEY"

qwen:
  model_name: "qwen3-vl-235b-a22b-instruct"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key: "YOUR_KEY"
```

## 🚀 Usage

The pipeline includes three steps: Generation, Evaluation, and Metric Calculation.

**1. Generate Images**
```bash
# bash scripts/run_generate.sh <generate_model> <prompts_json>
bash scripts/run_generate.sh gemini data/benchmark_demo.json
```

**2. Evaluate Safety (with Qwen-VL)**
```bash
# bash scripts/run_evaluate.sh <generate_model>
bash scripts/run_evaluate.sh gemini
```

**3. Calculate Metrics**
```bash
# bash scripts/run_metric.sh <generate_model>
bash scripts/run_metric.sh gemini
```

## 🚧 Data Availability
The full **Regulatory Compliance Benchmark** (prompts) and the prompt construction method are currently unpublished. They will be released following the paper's official publication.
