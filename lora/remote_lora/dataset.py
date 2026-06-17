import hashlib
import json
import os

from datasets import Dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from utils.logger import get_logger

logger = get_logger()


def _cache_signature(dataset_file: str, image_dir: str, tokenizer_name_or_path: str, max_seq_len: int) -> str:
    """
    根据影响数据集内容的关键参数生成一个确定性哈希签名。
    只要以下任意一项发生变化，签名就会变化，从而触发重新预处理：
    - dataset_file 的文件大小和最后修改时间（mtime）
    - image_dir 的绝对路径
    - tokenizer_name_or_path
    - max_seq_len
    """
    stat = os.stat(dataset_file)
    raw = "|".join([
        str(stat.st_size),
        str(stat.st_mtime),
        os.path.abspath(image_dir),
        str(tokenizer_name_or_path),
        str(max_seq_len),
    ])
    return hashlib.md5(raw.encode()).hexdigest()[:16]

SYSTEM_PROMPT = """
你是专业的多模态知识图谱构建专家，请根据输入文本与图像完成统一多模态关系抽取（UMRE）任务。

步骤：

1. 提取文本实体，格式：[TXT: 实体]；
2. 提取图像实体，每个图像实体为包含 name、caption、pos 字段的对象：
   - name：短标识符，格式 [IMG: 01]（编号递增）；
   - caption：该图像区域的内容描述；
   - pos：归一化 box 坐标 [x1, y1, x2, y2]。
3. 识别文本-文本、图像-图像、文本-图像之间的关系（用 name 引用图像实体）。

关系必须严格从以下集合选择：
self；
/per/per/{partner,relatives,opponent,alumni}；
/per/org/{opposed_to,leader_of,member_of}；
/per/loc/{place_of_birth,place_of_residence,place_of_governance}；
/per/misc/{president,awarded,religion,nationality,party,present_in}；
/org/org/{subsidiary,contain}；
/org/loc/locate_of；
/org/misc/{present_in,held_on}；
/loc/loc/contain；
/misc/loc/held_on。

其中 self 表示文本实体与图像实体指向同一现实对象；证据不足时不建立关系。

仅输出合法 JSON，不要输出解释性文字：

{
    "text_entities": [
        "[TXT: 实体1]",
        "[TXT: 实体2]"
    ],
    "image_entities": [
        {"name": "[IMG: 01]", "caption": "图像区域描述文字", "pos": [0.265, 0.41, 0.595, 0.99]},
        {"name": "[IMG: 02]", "caption": "另一图像区域描述文字", "pos": [0.10, 0.20, 0.50, 0.80]}
    ],
    "relations": [
        {
            "head": "[TXT: 实体1]",
            "relation": "/per/org/member_of",
            "tail": "[TXT: 实体2]"
        },
        {
            "head": "[TXT: 实体1]",
            "relation": "self",
            "tail": "[IMG: 01]"
        }
    ]
}
"""

USER_PROMPT = """
请根据以下图文内容完成统一多模态关系抽取任务。

文本内容：
{text}
"""


def _build_gt(doc: dict) -> dict:
    """
    将 Remote 数据集单条记录的 entity/rel 字段转换为结构化 ground-truth 字典。
    relation 字符串直接取自数据文件中 rel[].relation 字段，无需额外映射。

    返回格式：
    {
        "text_entities": ["[TXT: Pablo Casado]", ...],
        "image_entities": [
            {"name": "[IMG: 01]", "caption": "图像描述文字", "pos": [0.80, 0.79, 0.95, 0.99]},
            ...
        ],
        "relations": [
            {"head": "[TXT: ...]", "relation": "...", "tail": "[IMG: 01]"},
            ...
        ]
    }
    """
    entities = doc.get("entity", [])
    rels = doc.get("rel", [])

    # ---- 构建实体列表 ----
    text_entities = []
    image_entities = []

    # 图像实体需要编号，所以先做两遍扫描
    img_counter = 1
    entity_repr = []  # 每个实体对应的字符串表示，与 entities 索引对齐
    for ent in entities:
        if ent["type"] == "text":
            label = f"[TXT: {ent['text']}]"
            text_entities.append(label)
        else:  # image
            # pos 格式：[x1, y1, x2, y2, filename]，只取前4个 box 坐标
            pos = ent.get("pos", [])
            box_coords = [round(float(v), 5) for v in pos[:4]] if len(pos) >= 4 else []
            # name 使用短标识符，在 relations 中作为引用键
            name = f"[IMG: {img_counter:02d}]"
            image_entities.append({
                "name": name,
                "caption": ent["text"],
                "pos": box_coords,
            })
            # entity_repr 存短 name，供 relations 引用
            label = name
            img_counter += 1
        entity_repr.append(label)

    # ---- 构建关系列表 ----
    relations = []
    for rel in rels:
        head_idx = rel["head"]
        tail_idx = rel["tail"]
        relation = rel["relation"]
        # 过滤掉 "none" 关系（证据不足）
        if relation.lower() == "none":
            continue
        if head_idx < len(entity_repr) and tail_idx < len(entity_repr):
            relations.append({
                "head": entity_repr[head_idx],
                "relation": relation,
                "tail": entity_repr[tail_idx],
            })

    return {
        "text_entities": text_entities,
        "image_entities": image_entities,
        "relations": relations,
    }


def create_remote_dataset(
        dataset_file: str,
        image_dir: str,
        tokenizer_name_or_path: str,
        max_seq_len: int,
        num_proc: int = 16,      # 控制并行多进程的数量，建议设置为 CPU 核心数
        cache_dir: str = None,   # 缓存目录路径，None 表示不启用缓存
):
    """
    将 Remote 多模态实体-关系数据集转换为可直接用于 HF Trainer 训练的 Dataset。

    参数
    ----
    dataset_file : str
        JSON 文件路径，内容为 List[Dict]，每条数据格式见 README。
        rel[].relation 字段已是最终关系字符串（如 "/per/per/partner"、"none"），直接使用。
    image_dir : str
        图像文件所在目录，文件名来自 doc["image_id"]。
    tokenizer_name_or_path : str
        Qwen-VL 模型路径或 HuggingFace Hub ID，用于加载 AutoProcessor。
    max_seq_len : int
        最大序列长度，超出后截断。
    num_proc : int
        Dataset.map 并行进程数。
    cache_dir : str, optional
        预处理结果的缓存目录路径（HF Arrow 格式，通过 save_to_disk 写出）。
        - 若指定且目录存在且签名匹配，则直接 load_from_disk 加载，跳过全部预处理。
        - 若指定但目录不存在或签名不匹配，则正常预处理后通过 save_to_disk 写入。
        - 若为 None，则禁用缓存。
        注意：与 torch.save 不同，save_to_disk 基于 Arrow 内存映射，流式分块
        写出，不会将全量数据一次性载入内存，适用于含 pixel_values 的大型多模态数据集。

    返回
    ----
    datasets.Dataset
        包含 input_ids / attention_mask / labels 三列的 HF Dataset，
        labels 中 prompt 部分已被 mask 为 -100（仅对 assistant 回答计算 loss）。
    """

    # ------------------------------------------------------------------
    # 0. 缓存命中检查（仅当 cache_dir 已指定时）
    # ------------------------------------------------------------------
    current_sig = _cache_signature(dataset_file, image_dir, tokenizer_name_or_path, max_seq_len)
    _sig_file = os.path.join(cache_dir, ".cache_sig") if cache_dir else None
    if cache_dir is not None and os.path.isdir(cache_dir) and os.path.isfile(_sig_file):
        logger.info(f"发现缓存目录: {cache_dir}，正在验证签名...")
        try:
            with open(_sig_file, "r") as _f:
                saved_sig = _f.read().strip()
            if saved_sig == current_sig:
                logger.info("缓存签名匹配，直接 load_from_disk 加载缓存数据集，跳过预处理。")
                return load_from_disk(cache_dir)
            else:
                logger.warning("缓存签名不匹配（数据源或参数已变化），将重新预处理并覆盖缓存。")
        except Exception as e:
            logger.warning(f"缓存读取失败（{e}），将重新预处理。")

    # ------------------------------------------------------------------
    # 1. 读取原始数据，构建 messages 文本和 ground-truth 字符串
    #    （纯 Python 操作，单线程即可）
    # ------------------------------------------------------------------
    logger.info(f"正在读取数据集文件: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 用于传给 Dataset.from_dict 的中间列表
    all_messages: list[list[dict]] = []   # 每条样本的 messages（不含 assistant 回复）
    all_targets: list[str] = []           # 每条样本的 ground-truth JSON 字符串
    all_image_paths: list[str] = []       # 每条样本的图像本地路径

    skipped = 0
    for doc in tqdm(raw_data, desc="构建 messages & targets"):
        image_id = doc.get("image_id", "")
        image_path = os.path.join(image_dir, image_id)

        # 如果图像文件不存在则跳过，避免训练时崩溃
        if not os.path.isfile(image_path):
            logger.warning(f"图像文件不存在，跳过样本 id={doc.get('id', '?')}: {image_path}")
            skipped += 1
            continue

        text = doc.get("text", "")
        gt_dict = _build_gt(doc)
        target_str = json.dumps(gt_dict, ensure_ascii=False)

        # 按 README 规定的 message 格式构造输入
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT.strip()}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(text=text).strip()},
                    {"type": "image", "image": image_path},
                ],
            },
        ]

        all_messages.append(messages)
        all_targets.append(target_str)
        all_image_paths.append(image_path)

    if skipped:
        logger.warning(f"共跳过 {skipped} 条图像缺失样本。")
    logger.info(f"有效样本数: {len(all_messages)}")

    # ------------------------------------------------------------------
    # 3. 构建中间 HF Dataset（存储可序列化的字段）
    # ------------------------------------------------------------------
    # messages 列表作为 JSON 字符串存储，以便 Dataset.map 跨进程传递
    raw_dataset = Dataset.from_dict({
        "messages_json": [json.dumps(m, ensure_ascii=False) for m in all_messages],
        "target": all_targets,
        "image_path": all_image_paths,
    })

    # ------------------------------------------------------------------
    # 4. 加载 Processor（只加载一次）
    # ------------------------------------------------------------------
    logger.info(f"正在加载 Processor: {tokenizer_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    # 部分模型 pad_token 未设置
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # 5. 定义分词 & 特征提取函数（模仿 DOCRED preprocess_dataset.py 中的 tokenize_fn）
    # ------------------------------------------------------------------
    def tokenize_fn(example):
        messages: list[dict] = json.loads(example["messages_json"])
        target: str = example["target"]
        image_path: str = example["image_path"]

        # 将 assistant 回复拼入完整 messages
        full_messages = messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }
        ]

        # 加载图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"图像加载失败 ({image_path}): {e}，将跳过该样本（labels 全 -100）。")
            image = None

        # ---------- 计算 prompt 长度（用于构造 labels mask）----------
        # 仅用 system + user messages，开启 add_generation_prompt
        prompt_only_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,          # 先得到纯文本，再统一 tokenize
        )

        # ---------- 完整序列（含 assistant 回复）--------------------
        full_text = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        # ---------- Tokenize（带图像特征）----------------------------
        images_list = [image] if image is not None else []

        # 仅 prompt 部分
        prompt_enc = processor(
            text=prompt_only_text,
            images=images_list if images_list else None,
            return_tensors=None,        # 返回 Python list
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            # enable_thinking=False
        )
        prompt_len = len(prompt_enc["input_ids"])

        # 完整序列
        full_enc = processor(
            text=full_text,
            images=images_list if images_list else None,
            return_tensors=None,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            # enable_thinking=False
        )

        input_ids = full_enc["input_ids"][0] if isinstance(full_enc["input_ids"][0], list) else full_enc["input_ids"]
        attn_mask = full_enc["attention_mask"][0] if isinstance(full_enc["attention_mask"][0], list) else full_enc["attention_mask"]
        actual_valid_tokens = sum(attn_mask)

        # ---------- 构造 labels（mask 掉 prompt 部分）---------------
        if actual_valid_tokens <= prompt_len:
            # 极端情况：截断后 answer 部分完全消失
            labels = [-100] * len(input_ids)
        else:
            nums_answer_token = actual_valid_tokens - prompt_len
            nums_padding_token = len(input_ids) - actual_valid_tokens
            labels = (
                [-100] * prompt_len
                + input_ids[prompt_len: prompt_len + nums_answer_token]
                + [-100] * nums_padding_token
            )

        result = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

        # 部分 VL Processor 还会输出 pixel_values / image_grid_thw 等视觉特征列
        # 如果存在则一并保留，以供多模态模型使用
        for key in ("pixel_values", "image_grid_thw"):
            if key in full_enc:
                result[key] = full_enc[key]

        return result

    # ------------------------------------------------------------------
    # 6. 多进程并行 Tokenize
    # ------------------------------------------------------------------
    logger.info(f"启动多进程 ({num_proc} 进程) 分词 & 特征提取...")

    # 确定需要删除的中间列
    remove_cols = ["messages_json", "target", "image_path"]

    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        num_proc=num_proc,
        remove_columns=remove_cols,
        desc="Tokenizing & extracting visual features",
    )

    # ------------------------------------------------------------------
    # 7. 写入缓存（如果指定了 cache_dir）
    # ------------------------------------------------------------------
    if cache_dir is not None:
        logger.info(f"正在通过 save_to_disk 将预处理结果写入缓存目录: {cache_dir}")
        logger.info("（Arrow 格式，流式分块写出，不会一次性将全量数据载入内存）")
        os.makedirs(cache_dir, exist_ok=True)
        tokenized_dataset.save_to_disk(cache_dir)
        # 签名写入同目录下的隐藏文件
        with open(os.path.join(cache_dir, ".cache_sig"), "w") as _f:
            _f.write(current_sig)
        logger.info(f"缓存写入完成，共 {len(tokenized_dataset)} 条样本。")

    return tokenized_dataset