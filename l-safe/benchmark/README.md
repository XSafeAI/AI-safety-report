# LLM Safety Evaluation Framework

A modular, high-performance framework for evaluating the safety, robustness, and bias of Large Language Models (LLMs). This tool supports multi-threaded generation via OpenAI-compatible APIs and automated judging using **Qwen3Guard**.

## 🚀 Features

*   **Multi-Benchmark Support**: Integrated support for StrongReject, SorryBench, ALERT, Flames, and BBQ.
*   **High Concurrency**: Uses `ThreadPoolExecutor` for fast parallel generation via APIs.
*   **Automated Judging**: 
    *   Uses **Qwen/Qwen3Guard-Gen-8B** (local GPU) for safety scoring.
    *   Uses rule-based evaluation for BBQ (Bias Benchmark for QA).
*   **Modular Design**: Easy to extend with new datasets or models.
*   **Resumable**: Saves results incrementally; can skip generation to run judging on existing logs.

## 📂 Project Structure

```text
safety-eval/
├── data/                      # Place your benchmark datasets here
│   ├── alert_adversarial_100.jsonl
│   ├── bbq_100.jsonl
│   ├── Flames_1k_Chinese_100.jsonl
│   ├── sorry_bench_202503.csv
│   └── strongreject_small_dataset.csv
├── results/                   # Output logs and evaluation results
├── src/                       # Source code
│   ├── benchmarks/            # Individual benchmark logic
│   ├── judge.py               # Qwen3Guard implementation
│   └── utils.py               # Shared utilities (API, logging, IO)
├── main.py                    # Entry point script
├── requirements.txt           # Dependencies
└── .env                       # API Credentials
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/xingjunm/AI-safety-report.git
    cd l-safe/benchmark
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: To run the Qwen3Guard judge, you need a GPU and PyTorch installed.*

3.  **Prepare Data**:
    Place your dataset files in the `data/` folder. Ensure filenames match those defined in `main.py` (or update the paths in the code).

## ⚙️ Configuration

Create a `.env` file in the root directory to store your API credentials securely. **Do not commit this file.**

```ini
# .env
OPENAI_BASE_URL=your_actual_base_url_here
OPENAI_API_KEY=your_api_key_here
```

## 🏃 Usage

The `main.py` script is the single entry point for the framework.

### Basic Command
Run all benchmarks for a specific model:

```bash
python main.py --model grok-4-1-fast-non-reasoning
```

### Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--model` | **Required.** The model name to evaluate (as expected by the API). | `None` |
| `--bench` | Specific benchmark to run (`strongreject`, `sorry`, `alert`, `flames`, `bbq`, or `all`). | `all` |
| `--workers` | Number of concurrent API threads. | `10` |
| `--skip-gen` | Skip API generation phase (run judge on existing files). | `False` |
| `--skip-judge` | Skip the evaluation/judging phase. | `False` |

### Examples

**1. Run only the 'StrongReject' benchmark with 20 threads:**
```bash
python main.py --model grok-beta --bench strongreject --workers 20
```

**2. Run generation only (no GPU required for this step):**
```bash
python main.py --model deepseek-chat --bench all --skip-judge
```

**3. Run judging only (on previously generated files):**
*This step requires a GPU to load Qwen3Guard.*
```bash
python main.py --model grok-beta --bench all --skip-gen
```

## 📊 Supported Benchmarks

| Benchmark | Type | Evaluation Method | Description |
| :--- | :--- | :--- | :--- |
| **StrongReject** | Jailbreak | **Qwen3Guard** | Tests resilience against "Machiavellian" style prompts and forbidden topics. |
| **SorryBench** | Refusal | **Qwen3Guard** | Evaluates if the model appropriately refuses harmful requests (45+ topics). |
| **ALERT** | Red Teaming | **Qwen3Guard** | Adversarial prompts focusing on safety policies and risk categories. |
| **Flames** | Safety (CN) | **Qwen3Guard** | A comprehensive Chinese safety benchmark. |
| **BBQ** | Bias | **Exact Match** | (Bias Benchmark for QA) Multiple-choice questions to test social biases. |

## ⚖️ The Judge (Qwen3Guard)

For StrongReject, SorryBench, ALERT, and Flames, we use [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B).

*   The system loads the model locally using `transformers`.
*   It analyzes the prompt and the model's response.
*   It outputs a safety label (`Safe`, `Unsafe`, `Controversial`) and categorizes the content (e.g., `Violence`, `Illegal Acts`).
*   Results are appended to the JSONL output file under the `judge_result` key.