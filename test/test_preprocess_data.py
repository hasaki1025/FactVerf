"""
为 lora/preprocess_dataset.py 编写的 pytest 测试用例。
测试数据: test/test_data/test_dataset.json
映射文件: test/test_data/types.json
"""

import json
import os
import sys
import pytest

# 将项目根目录加入 sys.path，使 lora 包可以被导入
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from lora.preprocess_dataset import create_docred_hf_dataset

# ── 测试数据路径 ────────────────────────────────────────────────────────────────
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TEST_DATASET_JSON = os.path.join(TEST_DATA_DIR, "test_dataset.json")
TYPES_JSON = os.path.join(TEST_DATA_DIR, "types.json")


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_data():
    """返回原始 JSON 数据，供各测试参考。"""
    with open(TEST_DATASET_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def types_mapping():
    """返回 types.json 中的 relations 映射。"""
    with open(TYPES_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["relations"]


@pytest.fixture(scope="module")
def dataset_with_mapping():
    """使用映射文件生成的 Hugging Face Dataset。"""
    return create_docred_hf_dataset(TEST_DATASET_JSON, TYPES_JSON)


@pytest.fixture(scope="module")
def dataset_without_mapping():
    """不使用映射文件生成的 Hugging Face Dataset（关系字段保留原始 ID）。"""
    return create_docred_hf_dataset(TEST_DATASET_JSON)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 基本返回值校验
# ══════════════════════════════════════════════════════════════════════════════

class TestReturnType:
    """验证函数返回的对象类型及基本结构。"""

    def test_returns_dataset_object(self, dataset_with_mapping):
        from datasets import Dataset
        assert isinstance(dataset_with_mapping, Dataset), \
            "返回值应为 datasets.Dataset 类型"

    def test_dataset_has_messages_column(self, dataset_with_mapping):
        assert "messages" in dataset_with_mapping.column_names, \
            "Dataset 应包含 'messages' 列"

    def test_dataset_row_count_matches_raw(self, dataset_with_mapping, raw_data):
        assert len(dataset_with_mapping) == len(raw_data), \
            "Dataset 行数应与原始 JSON 文档数量一致"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 单条样本结构校验
# ══════════════════════════════════════════════════════════════════════════════

class TestSampleStructure:
    """校验每条样本（消息列表）的格式。"""

    def test_each_sample_has_two_messages(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            msgs = sample["messages"]
            assert len(msgs) == 2, \
                f"第 {i} 条样本应包含 2 条消息（user + assistant），实际为 {len(msgs)}"

    def test_first_message_is_user(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            assert sample["messages"][0]["role"] == "user", \
                f"第 {i} 条样本的第 1 条消息角色应为 'user'"

    def test_second_message_is_assistant(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            assert sample["messages"][1]["role"] == "assistant", \
                f"第 {i} 条样本的第 2 条消息角色应为 'assistant'"

    def test_user_message_contains_document_keyword(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            content = sample["messages"][0]["content"]
            assert "Document:" in content, \
                f"第 {i} 条样本的 user 消息应包含 'Document:'"




# ══════════════════════════════════════════════════════════════════════════════
# 3. assistant 输出（三元组）结构校验
# ══════════════════════════════════════════════════════════════════════════════

TRIPLET_REQUIRED_KEYS = {"head", "relation", "tail"}


class TestTripletStructure:
    """校验 assistant 回复中三元组的字段。"""

    def test_assistant_content_is_valid_json(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            content = sample["messages"][1]["content"]
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                pytest.fail(f"第 {i} 条样本 assistant 内容不是合法 JSON：{e}")

    def test_assistant_content_is_list(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            triplets = json.loads(sample["messages"][1]["content"])
            assert isinstance(triplets, list), \
                f"第 {i} 条样本 assistant 内容应为 JSON 数组"

    def test_each_triplet_has_required_keys(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            triplets = json.loads(sample["messages"][1]["content"])
            for j, triplet in enumerate(triplets):
                missing = TRIPLET_REQUIRED_KEYS - triplet.keys()
                assert not missing, \
                    f"第 {i} 条样本第 {j} 个三元组缺少字段：{missing}"

    def test_triplet_fields_are_non_empty_strings(self, dataset_with_mapping):
        for i, sample in enumerate(dataset_with_mapping):
            triplets = json.loads(sample["messages"][1]["content"])
            for j, triplet in enumerate(triplets):
                for key in TRIPLET_REQUIRED_KEYS:
                    val = triplet[key]
                    assert isinstance(val, str) and val.strip(), \
                        f"第 {i} 条样本第 {j} 个三元组字段 '{key}' 不应为空字符串"


# ══════════════════════════════════════════════════════════════════════════════
# 4. 关系 ID → verbose 名称映射校验
# ══════════════════════════════════════════════════════════════════════════════

class TestRelationMapping:
    """验证关系 ID 已被正确替换为 verbose 名称。"""

    def test_relation_is_not_raw_pid(self, dataset_with_mapping, raw_data, types_mapping):
        """使用映射文件时，关系字段不应保留原始 P-ID（若映射表中存在该 ID）。"""
        used_rel_ids = {
            label["r"]
            for doc in raw_data
            for label in doc.get("labels", [])
        }
        for i, sample in enumerate(dataset_with_mapping):
            triplets = json.loads(sample["messages"][1]["content"])
            for triplet in triplets:
                rel = triplet["relation"]
                for rel_id in used_rel_ids:
                    if rel_id in types_mapping and rel == rel_id:
                        expected_verbose = types_mapping[rel_id]["verbose"]
                        pytest.fail(
                            f"第 {i} 条样本关系字段保留了原始 ID '{rel_id}'，"
                            f"应为 '{expected_verbose}'"
                        )

    def test_known_relation_verbose_in_output(self, dataset_with_mapping, types_mapping):
        """验证 P17 → 'country' 的 verbose 名称出现在输出中。"""
        target_verbose = types_mapping["P17"]["verbose"]  # "country"
        all_relations = []
        for sample in dataset_with_mapping:
            triplets = json.loads(sample["messages"][1]["content"])
            all_relations.extend(t["relation"] for t in triplets)
        assert target_verbose in all_relations, \
            f"预期在输出中找到关系 verbose 名称 '{target_verbose}'，但未找到"


# ══════════════════════════════════════════════════════════════════════════════
# 5. 不使用映射文件时的行为
# ══════════════════════════════════════════════════════════════════════════════

class TestNoMapping:
    """当不传入映射文件时，关系字段应保留原始 P-ID。"""

    def test_relation_is_raw_pid_without_mapping(self, dataset_without_mapping, raw_data):
        used_rel_ids = {
            label["r"]
            for doc in raw_data
            for label in doc.get("labels", [])
        }
        for i, sample in enumerate(dataset_without_mapping):
            triplets = json.loads(sample["messages"][1]["content"])
            for triplet in triplets:
                assert triplet["relation"] in used_rel_ids, \
                    f"第 {i} 条样本：无映射时关系字段 '{triplet['relation']}' 应为原始 P-ID"


# ══════════════════════════════════════════════════════════════════════════════
# 6. 文本重建校验
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentReconstruction:
    """校验文档文本由句子 token 拼接而成。"""

    def test_document_text_contains_expected_tokens(self, dataset_with_mapping, raw_data):
        """第 0 条文档的 user 消息中应包含原始 token。"""
        doc = raw_data[0]
        # 取第一个句子的前几个 token 作为检查点
        first_sent_tokens = doc["sents"][0][:3]
        user_content = dataset_with_mapping[0]["messages"][0]["content"]
        for token in first_sent_tokens:
            assert token in user_content, \
                f"user 消息中未找到预期 token '{token}'"

    def test_document_text_contains_title(self, dataset_with_mapping, raw_data):
        """文档标题中的关键词应出现在 user 消息正文中。"""
        doc = raw_data[0]
        title_word = doc["title"].split()[0]  # "Skai"
        user_content = dataset_with_mapping[0]["messages"][0]["content"]
        assert title_word in user_content, \
            f"user 消息中未找到文档标题关键词 '{title_word}'"


# ══════════════════════════════════════════════════════════════════════════════
# 7. 错误处理：映射文件不存在
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingMappingFile:
    """当映射文件路径不存在时，函数应正常降级（使用原始 ID）。"""

    def test_nonexistent_mapping_file_does_not_raise(self):
        """传入不存在的映射文件路径时，函数不应抛出异常。"""
        ds = create_docred_hf_dataset(TEST_DATASET_JSON, "nonexistent_types.json")
        assert ds is not None
        assert len(ds) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. 错误处理：映射文件缺少 "relations" 键
# ══════════════════════════════════════════════════════════════════════════════

class TestInvalidMappingFile:
    """当映射文件缺少 'relations' 键时，函数应抛出 KeyError。"""

    def test_missing_relations_key_raises_key_error(self, tmp_path):
        bad_types = tmp_path / "bad_types.json"
        bad_types.write_text(json.dumps({"entities": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="relations"):
            create_docred_hf_dataset(TEST_DATASET_JSON, str(bad_types))


# ══════════════════════════════════════════════════════════════════════════════
# 9. 无 labels 字段的文档
# ══════════════════════════════════════════════════════════════════════════════

class TestNoLabels:
    """文档中若无 labels 字段，assistant 应返回空数组。"""

    def test_document_without_labels_produces_empty_triplets(self, tmp_path):
        doc_without_labels = [
            {
                "vertexSet": [[{"name": "Test", "type": "ORG", "pos": [0, 1], "sent_id": 0}]],
                "sents": [["Test", "document", "."]],
                "title": "Test"
                # 故意缺省 "labels" 字段
            }
        ]
        data_file = tmp_path / "no_labels.json"
        data_file.write_text(json.dumps(doc_without_labels), encoding="utf-8")

        ds = create_docred_hf_dataset(str(data_file), TYPES_JSON)
        assert len(ds) == 1
        triplets = json.loads(ds[0]["messages"][1]["content"])
        assert triplets == [], \
            "无 labels 的文档应产生空三元组列表"


# ══════════════════════════════════════════════════════════════════════════════
# 10. 人工核查：打印处理后的 messages（pytest -s 时可见）
# ══════════════════════════════════════════════════════════════════════════════

def test_print_messages_for_human_inspection():
    """
    读取 test_data 中的数据，经 preprocess_dataset 处理后，
    将每条样本的 messages 打印到控制台，供人工核查。

    运行方式（显示 print 输出）：
        pytest test/test_preprocess_data.py::test_print_messages_for_human_inspection -s -v
    """
    ds = create_docred_hf_dataset(TEST_DATASET_JSON, TYPES_JSON)

    separator = "=" * 80
    print(f"\n{separator}")
    print(f"  共 {len(ds)} 条样本（来自 {os.path.basename(TEST_DATASET_JSON)}）")
    print(separator)

    for i, sample in enumerate(ds):
        messages = sample["messages"]
        print(f"\n{'─' * 80}")
        print(f"  【样本 {i}】")
        print(f"{'─' * 80}")

        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]

            # assistant 内容是 JSON，格式化展示三元组
            if role == "ASSISTANT":
                print(f"\n[{role}] ：{content}")
            else:
                # user prompt 较长，只打印前 300 字符
                preview = content
                print(f"\n[{role}]\n{preview}")

    print(f"\n{separator}\n")
