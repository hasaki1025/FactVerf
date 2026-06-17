import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any, Dict

from openai import OpenAI
from tqdm import tqdm

from lora.preprocess_dataset import PROMPT_TEMPLATE
from utils.logger import get_logger

logger = get_logger()

infer_prompt = """\
Please extract entities in the given text and the relations between entities.
Please return ONLY a valid JSON array, each element is {"head": "...", "relation": "...", "tail": "..."}. Do not include any explanations, thinking process, or markdown formatting. Start directly with `[` and end with `]`.
Example: [{"head": "Apple", "relation": "founded by", "tail": "Steve Jobs"}]

Document:
"""

# infer_prompt = PROMPT_TEMPLATE
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""



def load_task_file(task_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(task_file):
        return []
    with open(task_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


def save_task_file(task_file: str, task_list: List[Dict[str, Any]]):
    with open(task_file, 'w', encoding='utf-8') as f:
        for task in task_list:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')


def fetch_inference(client: OpenAI, prompt: str, model: str, max_tokens: int, temperature: float, timeout: float, max_retries: int = 3) -> Any:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            generated_text = response.choices[0].message.content
            
            generated_text = generated_text.strip()
            
            # 移除思考过程 (由于 <think> 在 prompt 中，模型输出可能只包含到 </think> 结尾的部分)
            if "</think>" in generated_text:
                generated_text = generated_text.split("</think>")[-1]
            
            # 移除模型可能输出的 markdown 包裹块
            if generated_text.startswith("```json"):
                generated_text = generated_text[7:]
            if generated_text.startswith("```"):
                generated_text = generated_text[3:]
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]
            generated_text = generated_text.strip()
            
            # 校验 JSON 格式
            parsed_json = json.loads(generated_text)
            # 根据 PROMPT_TEMPLATE 要求，输出应为 JSON 数组 (List)
            if not isinstance(parsed_json, list):
                raise ValueError("JSON format error: expected a list")
            return parsed_json

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt + 1}: JSON decode error: {e}. Output was: {generated_text}")
        except ValueError as e:
            logger.warning(f"Attempt {attempt + 1}: Validation error: {e}. Output was: {generated_text}")
        except Exception as e:
            # 捕获包括 openai API 的超时、连接等异常
            logger.warning(f"Attempt {attempt + 1}: Request error: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(2)

    return None


def process_task(task: Dict[str, Any], client: OpenAI, model: str, max_tokens: int, temperature: float, timeout: float) -> bool:
    # 断点续传，如果已经存在 extraction_result 则跳过
    if 'extraction_result' in task:
        return False
        
    text = task.get('text', '')
    prompt = f"{infer_prompt}\n{text}"
    
    result = fetch_inference(client, prompt, model, max_tokens, temperature, timeout)
    if result is not None:
        task['extraction_result'] = result
        return True
    return False


def main():
    # --task_file dataset/FEVER/train_claim.jsonl --batch_size 32 --max_tokens 512
    parser = argparse.ArgumentParser(description="Run batched inference using HF model and LoRA.")
    parser.add_argument('--task_file', type=str, required=True, help='in/output file')
    parser.add_argument('--model_server_url', type=str, default='http://127.0.0.1:8000/v1', required=False, help='model server url')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for concurrent inference')
    parser.add_argument('--model', type=str, default='docred_lora', help='Model name for API request')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--timeout', type=float, default=300.0, help='Timeout for API request in seconds')
    args = parser.parse_args()

    task_list = load_task_file(args.task_file)
    tasks_to_process = [task for task in task_list if 'extraction_result' not in task]
    
    logger.info(f"Total tasks: {len(task_list)}, pending tasks: {len(tasks_to_process)}")
    
    if not tasks_to_process:
        logger.info("All tasks already completed.")
        return

    # 初始化 OpenAI Client，vLLM 等兼容接口通常可以使用任意字符串作为 API Key
    client = OpenAI(base_url=args.model_server_url, api_key="EMPTY")
    models = client.models.list()
    print("✅ 连接成功！当前服务端可用的模型如下：")
    print("-" * 40)

    # 遍历并打印所有可用模型的 ID（名称）
    for model in models.data:
        print(f"📦 模型名称 (Model ID): {model.id}")
        # 如果你想看更详细的信息，可以打印 model.model_dump()

    print("-" * 40)
    logger.info(f'当前使用模型ID为 : {args.model}')
    # 使用 ThreadPoolExecutor 实现并发推理
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        # 提交所有待处理的推理任务
        future_to_task = {
            executor.submit(process_task, task, client, args.model, args.max_tokens, args.temperature, args.timeout): task 
            for task in tasks_to_process
        }
        
        batch_save_size = max(10, args.batch_size)
        completed_in_batch = 0
        
        # 使用 tqdm 监控进度条
        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_process), desc="Inferring"):
            try:
                success = future.result()
                if success:
                    completed_in_batch += 1
            except Exception as e:
                logger.error(f"Task generated an exception: {e}")
                
            # 每完成一定数量任务，保存一次断点
            if completed_in_batch >= batch_save_size:
                save_task_file(args.task_file, task_list)
                completed_in_batch = 0
                
    # 所有任务执行完毕后最后保存一次
    save_task_file(args.task_file, task_list)
    logger.info("Inference completed and results saved.")


if __name__ == "__main__":
    main()
