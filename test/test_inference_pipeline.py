"""
test_inference_pipeline.py

集成测试：验证完整的两阶段数据处理流水线：
  阶段1: prepare_fever_llm.process_dataset 生成 train_evidence / train_claims
  阶段2: lora/run_inference.py 的核心函数（Mock LLM）为每条记录添加 extraction_result
  验证: claims 中引用的每条 evidence 都能在最终文件中找到（含 extraction_result 字段）

Mock 策略：
  - 使用 unittest.mock.patch 替换 lora.run_inference.fetch_inference，
    返回固定的三元组列表，无需真实 LLM 服务
"""

import json
import sqlite3
import unicodedata
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from dataset.FEVER.prepare_fever_llm import process_dataset
from lora.run_inference import (
    load_task_file,
    save_task_file,
    process_task,
)

# ──────────────────────────────────────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
TRAIN_JSONL    = PROJECT_ROOT / 'dataset' / 'FEVER' / 'train.jsonl'
WIKI_PAGES_DIR = PROJECT_ROOT / 'dataset' / 'FEVER' / 'wiki-pages'
EXISTING_DB    = PROJECT_ROOT / 'dataset' / 'FEVER' / 'wiki.db'

pytestmark = pytest.mark.skipif(
    not TRAIN_JSONL.exists(),
    reason=f'原始数据不存在: {TRAIN_JSONL}',
)

# ──────────────────────────────────────────────────────────────────────────────
# Mock LLM 返回值：一个简单的三元组列表
# ──────────────────────────────────────────────────────────────────────────────
MOCK_EXTRACTION_RESULT = [
    {"head": "Entity_A", "relation": "related_to", "tail": "Entity_B"}
]


def mock_fetch_inference(client, prompt, model, max_tokens, temperature, timeout, max_retries=3):
    """Mock fetch_inference：直接返回固定三元组，不调用任何网络。"""
    return MOCK_EXTRACTION_RESULT


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def get_db_connection() -> sqlite3.Connection:
    """优先使用已有 wiki.db（只读），否则从 wiki-pages 构建。"""
    if EXISTING_DB.exists():
        return sqlite3.connect(f'file:{EXISTING_DB}?mode=ro', uri=True)
    raise FileNotFoundError(
        f'wiki.db 不存在: {EXISTING_DB}\n'
        '请先运行 prepare_fever_llm.build_db 构建数据库。'
    )


def run_mock_inference(task_file: str, batch_size: int = 64):
    """
    模拟 lora/run_inference.py main() 的核心逻辑，
    使用 mock_fetch_inference 替换真实 LLM 调用。

    参数与 main() 保持一致（task_file 原地覆盖写入）。
    """
    task_list = load_task_file(task_file)
    tasks_to_process = [t for t in task_list if 'extraction_result' not in t]

    if not tasks_to_process:
        return  # 断点续传：全部已完成

    mock_client = MagicMock()  # OpenAI client 占位，不会真正被调用

    with patch('lora.run_inference.fetch_inference', side_effect=mock_fetch_inference):
        with ThreadPoolExecutor(max_workers=min(batch_size, len(tasks_to_process))) as executor:
            future_to_task = {
                executor.submit(
                    process_task, task, mock_client,
                    'mock_model', 512, 0.0, 30.0
                ): task
                for task in tasks_to_process
            }
            completed = 0
            for future in as_completed(future_to_task):
                try:
                    success = future.result()
                    if success:
                        completed += 1
                except Exception as e:
                    pass  # 测试中忽略单条失败，汇总阶段再检查

    # 原地覆盖写回（与 run_inference.main() 一致）
    save_task_file(task_file, task_list)


def load_evidence_index(evidence_path: Path) -> dict:
    """
    加载 evidence 文件，返回：
      {(nfc_page_id, sentence_id): item_dict}
    """
    index = {}
    with open(evidence_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = item.get('page_id', '')
            sid = item.get('sentence_id')
            if pid and sid is not None:
                key = (unicodedata.normalize('NFC', pid), sid)
                index[key] = item
    return index


def collect_claim_evidence_refs(train_path: Path) -> dict:
    """
    返回 {claim_id: {(nfc_page_id, sentence_id), ...}} 所有引用。
    """
    refs = {}
    with open(train_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cid = data.get('id')
            refs[cid] = set()
            for ev_group in data.get('evidence', []):
                for ann in ev_group:
                    if len(ann) >= 4 and ann[2] is not None and ann[3] is not None:
                        refs[cid].add(
                            (unicodedata.normalize('NFC', ann[2]), ann[3])
                        )
    return refs


# ──────────────────────────────────────────────────────────────────────────────
# Fixture：运行完整两阶段流水线（class scope，只运行一次）
# ──────────────────────────────────────────────────────────────────────────────

class TestInferencePipeline:
    """
    验证 process_dataset → mock inference 两阶段流水线的完整性：
    - 所有 evidence 都能在输出文件中找到
    - 所有条目都带有 extraction_result 字段
    - 特殊 Unicode 字符（如 José_María_Chacón）不丢失
    """

    @pytest.fixture(scope='class')
    def pipeline_output(self, tmp_path_factory):
        """
        运行完整流水线，返回输出文件路径。
        scope='class'：所有测试方法共享同一份输出，避免重复运行。
        """
        tmp = tmp_path_factory.mktemp('inference_pipeline')
        evidence_file = str(tmp / 'train_evidence.jsonl')
        claims_file   = str(tmp / 'train_claims.jsonl')

        # ── 阶段1：generate evidence & claims ─────────────────────────────
        conn = get_db_connection()
        try:
            process_dataset(
                str(TRAIN_JSONL),
                evidence_file,
                claims_file,
                conn,
            )
        finally:
            conn.close()

        # ── 阶段2：Mock LLM inference（原地覆盖写入） ────────────────────
        run_mock_inference(evidence_file, batch_size=64)
        run_mock_inference(claims_file,   batch_size=64)

        return {
            'evidence_file': Path(evidence_file),
            'claims_file':   Path(claims_file),
            'tmp':           tmp,
        }

    # ── 测试1：输出文件存在且非空 ───────────────────────────────────────────

    def test_output_files_exist(self, pipeline_output):
        """两个输出文件应正常创建且非空。"""
        for key in ('evidence_file', 'claims_file'):
            p = pipeline_output[key]
            assert p.exists(),             f'{key} 文件不存在: {p}'
            assert p.stat().st_size > 0,   f'{key} 文件为空: {p}'

    # ── 测试2：每条 evidence 都有 extraction_result ────────────────────────

    def test_all_evidence_have_extraction_result(self, pipeline_output):
        """
        inference 阶段后，evidence 文件中每条记录都应包含 extraction_result 字段。
        """
        evidence_file = pipeline_output['evidence_file']
        missing = []
        with open(evidence_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if 'extraction_result' not in item:
                    missing.append((i, item.get('page_id'), item.get('sentence_id')))

        assert not missing, (
            f'有 {len(missing)} 条 evidence 缺少 extraction_result 字段\n'
            f'前5条: {missing[:5]}'
        )

    def test_all_claims_have_extraction_result(self, pipeline_output):
        """
        inference 阶段后，claims 文件中每条记录都应包含 extraction_result 字段。
        """
        claims_file = pipeline_output['claims_file']
        missing = []
        with open(claims_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if 'extraction_result' not in item:
                    missing.append((i, item.get('id')))

        assert not missing, (
            f'有 {len(missing)} 条 claim 缺少 extraction_result 字段\n'
            f'前5条: {missing[:5]}'
        )

    # ── 测试3：extraction_result 格式正确 ──────────────────────────────────

    def test_extraction_result_format(self, pipeline_output):
        """
        extraction_result 应为列表，每个元素包含 head/relation/tail 键。
        """
        evidence_file = pipeline_output['evidence_file']
        bad_format = []
        with open(evidence_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                result = item.get('extraction_result')
                if not isinstance(result, list):
                    bad_format.append((i, 'extraction_result 不是 list', result))
                    continue
                for j, triple in enumerate(result):
                    if not isinstance(triple, dict):
                        bad_format.append((i, f'第{j}个元素不是 dict', triple))
                    elif not all(k in triple for k in ('head', 'relation', 'tail')):
                        bad_format.append((i, f'第{j}个元素缺少必要键', triple))

        assert not bad_format, (
            f'extraction_result 格式错误，共 {len(bad_format)} 处\n'
            f'前5处: {bad_format[:5]}'
        )

    # ── 测试4：claims 中所有 evidence 引用都能找到 ─────────────────────────

    def test_all_claim_evidence_refs_present(self, pipeline_output):
        """
        【核心测试】train.jsonl 中每个 claim 引用的 evidence，
        都应出现在最终的 train_evidence.jsonl 中。
        """
        evidence_file = pipeline_output['evidence_file']
        evidence_index = load_evidence_index(evidence_file)
        claim_refs     = collect_claim_evidence_refs(TRAIN_JSONL)

        all_refs = {ref for refs in claim_refs.values() for ref in refs}
        missing  = all_refs - set(evidence_index.keys())

        if missing:
            # 诊断：哪些 claim 受影响
            affected_claims = {
                cid: refs & missing
                for cid, refs in claim_refs.items()
                if refs & missing
            }
            sample_missing = sorted(missing, key=str)[:20]
            detail = '\n'.join(
                f'  page_id={repr(r[0])}  sentence_id={r[1]}'
                f'  非ASCII={not r[0].isascii()}'
                for r in sample_missing
            )
            pytest.fail(
                f'共 {len(missing)} 条 evidence 引用缺失，'
                f'影响 {len(affected_claims)} 个 claim\n'
                f'缺失示例（前20条）:\n{detail}'
            )

    # ── 测试5：特殊 Unicode 字符 evidence 不丢失 ───────────────────────────

    def test_unicode_evidence_not_lost(self, pipeline_output):
        """
        含非 ASCII 字符的 evidence 条目，
        经过 inference 覆写后应仍然存在于输出文件中。
        """
        evidence_file = pipeline_output['evidence_file']
        evidence_index = load_evidence_index(evidence_file)
        claim_refs     = collect_claim_evidence_refs(TRAIN_JSONL)

        all_refs = {ref for refs in claim_refs.values() for ref in refs}
        special_refs = {r for r in all_refs if not r[0].isascii()}

        if not special_refs:
            pytest.skip('train.jsonl 中无含非 ASCII 字符的 evidence 引用')

        missing_special = special_refs - set(evidence_index.keys())
        assert not missing_special, (
            f'含特殊 Unicode 字符的 evidence 缺失 {len(missing_special)} 条\n'
            f'前10条:\n' +
            '\n'.join(
                f'  page_id={repr(r[0])}  sentence_id={r[1]}'
                for r in sorted(missing_special, key=str)[:10]
            )
        )

    # ── 测试6：José_María_Chacón 回归测试 ─────────────────────────────────

    def test_jose_maria_chacon_regression(self, pipeline_output):
        """
        回归测试：José_María_Chacón (sentence_id=0) 经过完整流水线后
        仍在输出文件中，且包含 extraction_result 字段。
        """
        evidence_file = pipeline_output['evidence_file']
        evidence_index = load_evidence_index(evidence_file)

        target = (unicodedata.normalize('NFC', 'José_María_Chacón'), 0)
        assert target in evidence_index, (
            f'回归失败：José_María_Chacón sentence_id=0 在最终输出中缺失\n'
            f'查找 key: {repr(target)}'
        )

        item = evidence_index[target]
        assert 'extraction_result' in item, (
            'José_María_Chacón 存在于输出文件，但缺少 extraction_result 字段'
        )

    # ── 测试7：inference 幂等性（断点续传不重复处理） ─────────────────────

    def test_inference_idempotent(self, pipeline_output):
        """
        对已有 extraction_result 的文件再次运行 inference，
        不应改变已有结果（断点续传机制验证）。
        """
        evidence_file = str(pipeline_output['evidence_file'])

        # 记录首次 inference 后的内容
        before = load_task_file(evidence_file)
        before_results = {
            (unicodedata.normalize('NFC', item.get('page_id', '')), item.get('sentence_id')): item.get('extraction_result')
            for item in before
        }

        # 再次运行 inference（应全部跳过，因为已有 extraction_result）
        run_mock_inference(evidence_file, batch_size=64)

        after = load_task_file(evidence_file)
        after_results = {
            (unicodedata.normalize('NFC', item.get('page_id', '')), item.get('sentence_id')): item.get('extraction_result')
            for item in after
        }

        # 数量不变
        assert len(before) == len(after), (
            f'重复 inference 后记录数变化: {len(before)} → {len(after)}'
        )

        # 每条结果不变
        changed = [
            k for k in before_results
            if before_results[k] != after_results.get(k)
        ]
        assert not changed, (
            f'断点续传失败：{len(changed)} 条记录的 extraction_result 被重复修改\n'
            f'前5条: {changed[:5]}'
        )

    # ── 测试8：原文本字段在 inference 后保持不变 ───────────────────────────

    def test_original_fields_preserved_after_inference(self, pipeline_output):
        """
        inference 覆写文件后，原有字段（page_id / sentence_id / text）
        应保持不变，不被 LLM 结果污染。
        """
        evidence_file = pipeline_output['evidence_file']
        required_fields = {'page_id', 'sentence_id', 'text', 'extraction_result'}
        bad_items = []

        with open(evidence_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                missing = required_fields - set(item.keys())
                if missing:
                    bad_items.append((i, item.get('page_id'), missing))

        assert not bad_items, (
            f'{len(bad_items)} 条记录缺少必要字段\n'
            f'前5条: {bad_items[:5]}'
        )

    # ── 测试9：GNN 预处理集成测试 (prepare_graph_data 逻辑验证) ──────────────

    def test_end_to_end_graph_data_preparation(self, pipeline_output):
        """
        终极集成测试：模拟 gnn.preprocess_dataset.prepare_graph_data 的核心逻辑，
        确保经过完整预处理和 inference 后的文件，能够被 GNN 的加载函数无缝使用，
        且不存在任何 `未找到 evidence_id` 的情况。
        """
        from gnn.preprocess_dataset import load_graph_data
        
        evidence_file = str(pipeline_output['evidence_file'])
        claims_file   = str(pipeline_output['claims_file'])
        
        # 调用与实际代码完全相同的加载函数
        claim_llm_rel, label_mapping, claim_evi_rel_mapping, evidence_llm_rel = (
            load_graph_data(str(TRAIN_JSONL), claims_file, evidence_file)
        )
        
        missing_evidences = []
        
        # 完全复制 prepare_graph_data 的逻辑
        for item in claim_llm_rel:
            claim_id = item['id']
            # label 校验
            if label_mapping.get(claim_id) is None:
                continue
                
            if 'extraction_result' not in item:
                continue
                
            for evidence_id in claim_evi_rel_mapping.get(claim_id, []):
                evidence_meta_data = evidence_llm_rel.get(evidence_id)
                if evidence_meta_data is None:
                    missing_evidences.append((claim_id, evidence_id))
                    
        assert not missing_evidences, (
            f'【终极检查失败】在 GNN 加载阶段仍然发现 {len(missing_evidences)} 处缺失的 evidence！\n'
            f'前 10 处缺失详情 (claim_id, evidence_id):\n' +
            '\n'.join(str(e) for e in missing_evidences[:10])
        )
