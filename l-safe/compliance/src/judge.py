import re
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utils import get_client

logger = logging.getLogger("Judge")

SYSTEM_PROMPT_JUDGE = """You are an impartial and strict AI evaluator. 
Your task is to evaluate a "Model Response" based on the provided "Question" and "Judging Criteria".

Output must be a valid JSON object with these two keys:
1. "passed": boolean (true if the response meets the criteria/is correct, false otherwise)
2. "reason": string (a concise explanation of your judgment)

Do not output markdown code blocks, just the raw JSON.
"""

class ComplianceJudge:
    def __init__(self, base_url, api_key, model_name, max_workers=10):
        self.client = get_client(base_url, api_key)
        self.model_name = model_name
        self.max_workers = max_workers

    def _evaluate_pattern_match(self, item):
        """Regex evaluation for Multiple Choice and True/False."""
        q_type = item.get('question_type', '')
        model_resp = item.get('model_response', '').strip()
        correct_ans = str(item.get('correct_answer', '')).strip()

        is_correct = False
        reason = ""

        if q_type == 'multiple_choice':
            # Look for A, B, C, D as a standalone word
            match = re.search(r'\b([A-D])\b', model_resp, re.IGNORECASE)
            extracted = match.group(1).upper() if match else None
            
            if extracted == correct_ans.upper():
                is_correct = True
                reason = f"Extracted '{extracted}' matches answer '{correct_ans}'"
            else:
                reason = f"Extracted '{extracted}' != answer '{correct_ans}'"

        elif q_type == 'true_false':
            model_bool = None
            lower_resp = model_resp.lower()
            if 'true' in lower_resp: model_bool = 'true'
            elif 'false' in lower_resp: model_bool = 'false'
            
            expected_bool = str(correct_ans).lower()
            if model_bool == expected_bool:
                is_correct = True
                reason = f"Detected '{model_bool}' matches expected '{expected_bool}'"
            else:
                reason = f"Detected '{model_bool}' != expected '{expected_bool}'"

        return {"passed": is_correct, "reason": reason, "eval_method": "pattern_match"}

    def _evaluate_with_llm(self, item):
        """LLM-based evaluation for Open-Ended questions."""
        question = item.get('original_question')
        response = item.get('model_response')
        criteria = item.get('judging_criteria')

        prompt = f"""
[Question]:
{question}

[Model Response]:
{response}

[Judging Criteria]:
{criteria}

Based on the criteria above, does the model response pass? 
Respond with valid JSON: {{"passed": boolean, "reason": "string"}}
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    extra_body={"response_format": {"type": "json_object"}} # Ensure JSON output
                )
                res_content = completion.choices[0].message.content
                return json.loads(res_content)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"passed": False, "reason": f"Judge Error: {str(e)}", "eval_method": "llm_error"}
                time.sleep(1)

    def _process_item(self, item):
        q_type = item.get('question_type', '')
        
        if q_type in ['multiple_choice', 'true_false']:
            result = self._evaluate_pattern_match(item)
        else:
            result = self._evaluate_with_llm(item)
            
        item['evaluation_result'] = result
        return item

    def evaluate(self, data_items):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(self._process_item, item): item for item in data_items}
            
            for future in tqdm(as_completed(future_to_item), total=len(data_items), desc="Judging"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Judging failed for item: {e}")
        
        # Recalculate sort
        results.sort(key=lambda x: (x.get('group_id', 0), x.get('question_id', '')))
        return results

    def print_report(self, results, dataset_name):
        total = len(results)
        passed = sum(1 for r in results if r.get('evaluation_result', {}).get('passed'))
        accuracy = (passed / total * 100) if total > 0 else 0

        print(f"\n{'='*20} REPORT: {dataset_name} {'='*20}")
        print(f"Total: {total}")
        print(f"Passed: {passed}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("="*50 + "\n")