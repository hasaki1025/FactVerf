import json

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """\
Please extract entities in the given text and the relations between entities. Let’s think step by step.
Please return in JSON form: a JSON array, each element is {"head": "...", "relation": "...", "tail": "..."}.
Example: [{"head": "Apple", "relation": "founded by", "tail": "Steve Jobs"}]

Document:
"""


def create_docred_hf_dataset(
        data_set_file: str,
        tokenizer_name_or_path: str,
        max_seq_len: int,
        rel_mapping_file: str = None,
        num_proc: int = 8  # 新增：控制并行多进程的数量，建议设置为 CPU 核心数
):
    # 1. 加载映射 (适配 types.json 的嵌套结构)
    rel_mapping = {}
    if rel_mapping_file:
        try:
            with open(rel_mapping_file, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)
                if "relations" not in raw_mapping:
                    raise KeyError('types.json 文件缺少 "relations" 键，请检查文件结构是否正确。')
                for rel_id, rel_data in raw_mapping["relations"].items():
                    rel_mapping[rel_id] = rel_data.get("verbose", rel_id)
        except FileNotFoundError:
            pass

    # 2. 加载并解析基础文本数据 (这一步很快，单线程即可)
    logger.info('正在解析 JSON 并构建纯文本 Prompt/Target...')
    with open(data_set_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    text_data = {"prompt": [], "target": []}
    for doc in tqdm(raw_data, desc="Building texts"):
        reconstructed_sentences = [" ".join(sent) for sent in doc['sents']]
        document_text = " ".join(reconstructed_sentences)

        gt_triplets = []
        for label in doc.get('labels', []):
            head_entity = doc['vertexSet'][label['h']][0]
            tail_entity = doc['vertexSet'][label['t']][0]
            rel_id = label['r']
            rel_text = rel_mapping.get(rel_id, rel_id)

            gt_triplets.append({
                "head": head_entity['name'],
                "relation": rel_text,
                "tail": tail_entity['name'],
            })

        prompt = PROMPT_TEMPLATE + '\n' + document_text
        target = json.dumps(gt_triplets, ensure_ascii=False)

        text_data["prompt"].append(prompt)
        text_data["target"].append(target)

    # 先构建未分词的 HF Dataset
    raw_dataset = Dataset.from_dict(text_data)

    # 3. 实例化 Tokenizer (核心修复：移出循环，只加载一次)
    logger.info(f'正在加载 Tokenizer: {tokenizer_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. 定义分词映射函数
    def tokenize_fn(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["target"]}
        ]

        user_prompt_dict = tokenizer.apply_chat_template(
            [messages[0]],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False
        )
        user_prompt_len = len(user_prompt_dict['input_ids'])

        result = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            max_length=max_seq_len,
            padding=False,
            truncation=True,
            return_dict=True,
            enable_thinking=False
        )

        input_ids = result['input_ids']
        attn_mask = result['attention_mask']

        actual_valid_tokens = sum(attn_mask)

        # 截断保护与 label 构建
        if actual_valid_tokens <= user_prompt_len:
            labels = [-100] * len(input_ids)
        else:
            nums_answer_token = actual_valid_tokens - user_prompt_len
            nums_padding_token = len(input_ids) - actual_valid_tokens
            labels = (
                    [-100] * user_prompt_len +
                    input_ids[user_prompt_len: user_prompt_len + nums_answer_token] +
                    [-100] * nums_padding_token
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels  # 注意：如果配合 HF Trainer，键名建议改为 "labels"
        }

    # 5. 多进程加速 Tokenize
    logger.info(f'启动多进程 ({num_proc} 进程) 分词处理...')
    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        num_proc=num_proc,  # 开启多进程并行
        remove_columns=["prompt", "target"],  # 清理中间变量，节省内存
        desc="Tokenizing dataset"
    )

    return tokenized_dataset