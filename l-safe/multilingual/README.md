# Multilingual Safety Evaluation

## How to run
1) Evaluation on PGP dataset:
   ```bash
   python test_PGP.py --model GPT # Evaluate GPT-5.2
   python test_PGP.py --model Gemini3 # Evaluate Gemini 3 Pro
   python test_PGP.py --model Grok # Evaluate Grok 4.1 Fast
   python test_PGP.py --model Qwen # Evaluate Qwen3-VL
   python test_PGP.py --model Doubao # Evaluate Doubao 1.6
   python test_PGP.py # Evaluate all supported models
   ```
2) Evaluation on ML-Bench dataset:
   ```bash
   python test_ML-Bench.py --model GPT # Evaluate GPT-5.2
   python test_ML-Bench.py --model Gemini3 # Evaluate Gemini 3 Pro
   python test_ML-Bench.py --model Grok # Evaluate Grok 4.1 Fast
   python test_ML-Bench.py --model Qwen # Evaluate Qwen3-VL
   python test_ML-Bench.py --model Doubao # Evaluate Doubao 1.6
   python test_ML-Bench.py # Evaluate all supported models
   ```
