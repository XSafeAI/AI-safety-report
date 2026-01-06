# T2I Adversarial Evaluation

Walkthrough of the adversarial image-generation evaluation pipeline.

## Overview
- Generate images from adversarial text prompts across multiple models.
- Score the generated images for nudity, violence, and hate using Grok (xAI).
- Aggregate safety metrics (toxic bypass rate, harmful/safe/refusal rates, and average toxicity) per attack method and category.

## Prerequisites
- Python 3.10+ recommended.
- Install dependencies:
  ```bash
  pip install openai json-repair google-genai requests
  ```

## Configure API keys
Update the placeholders in the code before running:
- `image_generation.py`: `GOOGLE_API_KEY` and `DOUBAO_API_KEY`.
- `grok_evaluator.py`: `XAI_API_KEY`.

## Data layout
- Prompts live under `data/` as CSV files: `{method}_{category}.csv`.
- Supported attack methods: `pgj`, `genbreak`.
- Supported categories: `nudity`, `violence`, `hate`.

## How to run
1) Generate images for a model (names come from `MODELS` in `image_generation.py`, e.g., `gemini-3-pro-image`):
   ```bash
   python image_generation.py --model gemini-3-pro-image
   ```
   - Images are saved to `generated_images/<model>/<method>/<category>/`.
   - A generation log is written to `evaluation_log_<model>.json`.

2) Evaluate toxicity of generated images with Grok:
   ```bash
   python eval_toxicity.py --model gemini-3-pro-image
   ```
   - Writes scores to `toxicity_evaluation_log_<model>.json`.

3) Calculate safety metrics:
   ```bash
   python calculate_metrics.py --model gemini-3-pro-image
   ```
   - Outputs `metrics_results_<model>.json` with aggregated per-method/per-category metrics.

## Notes
- Re-run steps with different `--model` values to compare providers.
- If a prompt already produced an image, generation is skipped and the existing file is reused.
- Rate limits trigger exponential backoff during image generation; failures are recorded in the logs.
