# LLM Compliance Evaluation Framework

A specialized framework for evaluating Large Language Models (LLMs) against major AI safety and compliance regulations. This tool automates the testing process using a hybrid evaluation approach (Regex Pattern Matching + LLM-based Judging) to ensure models adhere to standards such as the **EU AI Act**, **NIST AI RMF**, and **FEAT**.

## 🚀 Key Features

*   **Multi-Regulation Support**: Built-in support for multiple compliance datasets (EU AI Act, FEAT, NIST).
*   **Hybrid Judging Engine**:
    *   **Pattern Matching**: Accurate regex extraction for Multiple Choice and True/False questions.
    *   **LLM Judge**: Uses a high-performance LLM (e.g., Qwen-235B, GPT-4) to evaluate Open-Ended responses based on specific legal criteria.
*   **Data Handling**: Automatically flattens complex dataset structures (handling question variants) and filters out multi-modal tasks.
*   **High Concurrency**: efficient multi-threaded generation and evaluation.

## 📂 Project Structure

```text
compliance-eval/
├── data/                      # Regulation datasets (JSON)
│   ├── EU_AI_Act.json
│   ├── FEAT.json
│   └── NIST.json
├── results/                   # Output files (Generated responses & Judge results)
├── src/                       # Core logic
│   ├── generation.py          # API handling for the model being tested
│   ├── judge.py               # Hybrid evaluation logic
│   └── utils.py               # IO and Helper functions
├── main.py                    # Entry point script
├── requirements.txt           # Python dependencies
└── .env                       # API Configuration
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/xingjunm/AI-safety-report.git
    cd l-safe/compliance
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

Create a `.env` file in the root directory. You need to configure two separate endpoints: one for the **Model Under Test (Generator)** and one for the **Judge Model**.

```ini
# --- Generator Configuration (The model being tested) ---
GENERATOR_BASE_URL=your_base_url_here
GENERATOR_API_KEY=your_api_key_here

# --- Judge Configuration (Qwen3-235B or similar) ---
# If using local vLLM/SGLang:
JUDGE_BASE_URL=http://localhost:8000/v1
JUDGE_API_KEY=EMPTY
JUDGE_MODEL_NAME=your_judge_model_name_here
```

## 🏃 Usage

The `main.py` script handles both generation (getting answers from your model) and judging (scoring those answers).

### 1. Run Full Evaluation
This runs all datasets (`eu`, `feat`, `nist`), generates answers, and immediately scores them.

```bash
python main.py --model grok-4-1-fast-non-reasoning --dataset all --workers 10
```

### 2. Run Specific Dataset
Evaluate only the EU AI Act dataset:

```bash
python main.py --model grok-4-1-fast-non-reasoning --dataset eu
```

### 3. Run Judge Only
If you have already generated responses and want to re-run the scoring logic without calling the Generator API again:

```bash
python main.py --model grok-4-1-fast-non-reasoning --skip-gen
```

## 📊 Methodology

The framework applies different evaluation strategies based on the `question_type` in the dataset:

| Question Type | Evaluation Method | Logic |
| :--- | :--- | :--- |
| **Multiple Choice** | **Regex Pattern Match** | Extracts options (A, B, C, D) from the response and compares with the ground truth. |
| **True / False** | **Regex Pattern Match** | Detects boolean keywords (True/False) in the response. |
| **Open Ended** | **LLM Judge** | Sends the Question, Model Response, and specific `Judging Criteria` to the Judge LLM to determine if the response passes. |

## 📝 Datasets

The `data/` folder contains JSON files derived from regulatory documents:

*   **EU_AI_Act.json**: Questions based on the European Union Artificial Intelligence Act.
*   **NIST.json**: Questions based on the NIST AI Risk Management Framework.
*   **FEAT.json**: Questions based on Fairness, Ethics, Accountability, and Transparency principles.