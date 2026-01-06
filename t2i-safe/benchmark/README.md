# T2I Benchmark Evaluation

Here are scripts for batch requesting text-to-image generation models and evaluating their performance on T2I-Safety benchmark.

## Files
- `batch_req_gemini.py`: Script to batch request image generation from Gemini 3 Pro (Nano Banana Pro) model.
- `batch_req_seedream.py`: Script to batch request image generation from Seedream-4.5 model.
- `eavl.py`: Evaluation script to compute metrics on generated images.

## Usage

1. pip install necessary dependencies:
   ```bash
   pip install volcengine-python-sdk[ark] 
   pip install -q -U google-genai
   ```
2. Update API keys and configurations in `batch_req_gemini.py` and `batch_req_seedream.py`.
3. Run the batch request scripts to generate images.
4. Use `eavl.py` to evaluate the generated images against the T2I-Safety benchmark.
    ```bash
    python eavl.py --output_dir <generated_images_directory>
    ```
