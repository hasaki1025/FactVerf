"""
test_evidence_completeness.py

集成测试：验证 prepare_fever_llm.process_dataset 生成的 train_evidence.jsonl
覆盖了 train.jsonl 中所有 claims 所引用的 evidence 条目。

测试流程：
  1. 使用已有的 wiki.db（避免重建耗时），若不存在则调用 build_db 从 wiki-pages 构建
  2. 调用 process_dataset 将输出写入临时目录
  3. 解析 train.jsonl 中所有 (page_id, sentence_id) evidence 引用
  4. 检查每一条引用是否都在生成的 train_evidence.jsonl 中
  5. 报告缺失条目的详细信息（数量、Unicode 形式、是否因规范化导致匹配失败）
"""

import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest

from dataset.FEVER.prepare_fever_llm import build_db, process_dataset

# ──────────────────────────────────────────────────────────────────────────────
# 路径常量（相对于项目根目录运行 pytest）
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_JSONL   = PROJECT_ROOT / 'dataset' / 'FEVER' / 'train.jsonl'
WIKI_PAGES_DIR = PROJECT_ROOT / 'dataset' / 'FEVER' / 'wiki-pages'
# 优先使用已有的 wiki.db，避免重建（耗时数分钟）
EXISTING_DB   = PROJECT_ROOT / 'dataset' / 'FEVER' / 'wiki.db'


# ──────────────────────────────────────────────────────────────────────────────
# 跳过条件：原始数据文件不存在时跳过
# ──────────────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not TRAIN_JSONL.exists(),
    reason=f'原始数据文件不存在: {TRAIN_JSONL}'
)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def load_evidence_refs_from_train(train_path: Path) -> dict:
    """
    解析 train.jsonl，返回所有 claim 引用的 evidence。
    返回格式：
      {
        claim_id: {
          (nfc_page_id, sentence_id),   # NFC 规范化后的引用
          ...
        },
        ...
      }
    还返回原始（未规范化）引用集合，用于诊断编码问题。
    """
    refs_nfc = {}      # claim_id -> set of (nfc_page_id, sentence_id)
    refs_raw = {}      # claim_id -> set of (raw_page_id, sentence_id)

    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            claim_id = data.get('id')
            refs_nfc[claim_id] = set()
            refs_raw[claim_id] = set()

            for ev_group in data.get('evidence', []):
                for annotation in ev_group:
                    if len(annotation) >= 4:
                        page_id    = annotation[2]
                        sentence_id = annotation[3]
                        if page_id is not None and sentence_id is not None:
                            refs_raw[claim_id].add((page_id, sentence_id))
                            refs_nfc[claim_id].add(
                                (unicodedata.normalize('NFC', page_id), sentence_id)
                            )

    return refs_nfc, refs_raw


def load_generated_evidence(evidence_path: Path) -> set:
    """
    解析生成的 evidence 文件，返回所有 (nfc_page_id, sentence_id) 的集合。
    """
    generated = set()
    with open(evidence_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            page_id     = item.get('page_id', '')
            sentence_id = item.get('sentence_id')
            if page_id and sentence_id is not None:
                generated.add(
                    (unicodedata.normalize('NFC', page_id), sentence_id)
                )
    return generated


def get_or_build_db(tmp_path: Path) -> sqlite3.Connection:
    """
    若项目中已有 wiki.db，直接连接（只读）；否则从 wiki-pages 构建。
    返回 sqlite3 connection。
    """
    if EXISTING_DB.exists():
        # 直接连接已有 DB，以只读模式打开避免意外写入
        conn = sqlite3.connect(f'file:{EXISTING_DB}?mode=ro', uri=True)
        return conn
    else:
        db_path = str(tmp_path / 'wiki.db')
        print(f'\n未找到已有 wiki.db，从 wiki-pages 构建: {db_path}')
        conn = build_db(str(WIKI_PAGES_DIR), db_path)
        return conn


# ──────────────────────────────────────────────────────────────────────────────
# 测试类
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidenceCompleteness:
    """
    集成测试：验证 process_dataset 生成的 evidence 文件覆盖所有 claim 的引用。
    """

    @pytest.fixture(scope='class')
    def generated_files(self, tmp_path_factory):
        """
        Class-scope fixture：运行一次 process_dataset，供所有测试共用输出文件。
        """
        tmp_path = tmp_path_factory.mktemp('evidence_completeness')
        evidence_out = tmp_path / 'train_evidence.jsonl'
        claims_out   = tmp_path / 'train_claims.jsonl'

        conn = get_or_build_db(tmp_path)
        try:
            process_dataset(
                str(TRAIN_JSONL),
                str(evidence_out),
                str(claims_out),
                conn,
            )
        finally:
            conn.close()

        return {
            'evidence_out': evidence_out,
            'claims_out':   claims_out,
            'tmp_path':     tmp_path,
        }

    def test_output_files_created(self, generated_files):
        """验证输出文件被正常创建且非空。"""
        evidence_out = generated_files['evidence_out']
        claims_out   = generated_files['claims_out']

        assert evidence_out.exists(), f'evidence 输出文件未创建: {evidence_out}'
        assert claims_out.exists(),   f'claims 输出文件未创建: {claims_out}'
        assert evidence_out.stat().st_size > 0, 'evidence 文件为空'
        assert claims_out.stat().st_size > 0,   'claims 文件为空'

    def test_claims_count_matches(self, generated_files):
        """验证生成的 claims 数量与 train.jsonl 一致。"""
        claims_out = generated_files['claims_out']

        with open(TRAIN_JSONL, encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())

        with open(claims_out, encoding='utf-8') as f:
            claims_count = sum(1 for line in f if line.strip())

        assert claims_count == train_count, (
            f'claims 数量不一致: 预期 {train_count}，实际 {claims_count}'
        )

    def test_all_evidence_refs_found(self, generated_files):
        """
        【核心测试】验证 train.jsonl 中所有 claim 引用的 evidence 都出现在输出文件中。
        失败时输出详细的缺失条目信息，按 Unicode 形式分类诊断。
        """
        evidence_out = generated_files['evidence_out']

        refs_nfc, refs_raw = load_evidence_refs_from_train(TRAIN_JSONL)
        generated = load_generated_evidence(evidence_out)

        # 汇总所有 evidence 引用（去重）
        all_refs_nfc = set()
        all_refs_raw = {}
        for claim_id, refs in refs_nfc.items():
            all_refs_nfc.update(refs)
        for claim_id, refs in refs_raw.items():
            for raw_ref in refs:
                nfc_ref = (unicodedata.normalize('NFC', raw_ref[0]), raw_ref[1])
                all_refs_raw[nfc_ref] = raw_ref  # nfc_ref -> 原始 raw_ref

        missing = all_refs_nfc - generated

        if not missing:
            return  # 全部找到，测试通过

        # ── 构造详细的诊断报告 ──────────────────────────────────────────────
        report_lines = [
            f'\n【evidence 覆盖率检查失败】',
            f'总引用数（去重）: {len(all_refs_nfc)}',
            f'生成文件条目数:   {len(generated)}',
            f'缺失条目数:       {len(missing)}',
            '',
            '缺失条目详情（前 50 条）：',
        ]

        # 将缺失条目按 Unicode 问题分类
        nfc_nfd_mismatch = []   # 原始是 NFD，NFC 规范化后应能找到但仍缺失
        truly_missing    = []   # 原始就是 NFC，但在 wiki.db 中也不存在

        for nfc_ref in sorted(missing, key=lambda x: str(x))[:50]:
            raw_ref = all_refs_raw.get(nfc_ref, nfc_ref)
            raw_pid = raw_ref[0]
            nfc_pid = nfc_ref[0]
            is_nfd_orig = raw_pid != nfc_pid

            # 哪些 claim 引用了这条缺失 evidence
            citing_claims = [
                cid for cid, refs in refs_nfc.items()
                if nfc_ref in refs
            ]

            line = (
                f'  page_id={repr(nfc_pid)}'
                f'  sentence_id={nfc_ref[1]}'
                f'  原始NFD={is_nfd_orig}'
                f'  被{len(citing_claims)}个claim引用'
                f'  (示例claim_id={citing_claims[0] if citing_claims else "?"})'
            )
            report_lines.append(line)

            if is_nfd_orig:
                nfc_nfd_mismatch.append(nfc_ref)
            else:
                truly_missing.append(nfc_ref)

        report_lines += [
            '',
            f'原始为 NFD 形式（编码问题）: {len(nfc_nfd_mismatch)} 条',
            f'原始为 NFC 但仍缺失（wiki.db 无此数据）: {len(truly_missing)} 条',
        ]

        pytest.fail('\n'.join(report_lines))

    def test_no_unicode_encoding_loss(self, generated_files):
        """
        专项检查：验证含特殊 Unicode 字符的 evidence 没有因编码问题丢失。
        检测范围：page_id 含非 ASCII 字符的所有引用。
        """
        evidence_out = generated_files['evidence_out']
        refs_nfc, refs_raw = load_evidence_refs_from_train(TRAIN_JSONL)
        generated = load_generated_evidence(evidence_out)

        # 筛选出含特殊字符（非纯 ASCII）的引用
        special_refs = {
            nfc_ref
            for refs in refs_nfc.values()
            for nfc_ref in refs
            if not nfc_ref[0].isascii()
        }

        if not special_refs:
            pytest.skip('train.jsonl 中无含非 ASCII 字符的 evidence 引用')

        missing_special = special_refs - generated

        assert not missing_special, (
            f'含特殊 Unicode 字符的 evidence 缺失 {len(missing_special)} 条 '
            f'（共 {len(special_refs)} 条特殊字符引用）\n'
            f'缺失示例（前10条）:\n' +
            '\n'.join(
                f'  page_id={repr(r[0])}  sentence_id={r[1]}'
                for r in sorted(missing_special, key=lambda x: str(x))[:10]
            )
        )

    def test_jose_maria_chacon_evidence_present(self, generated_files):
        """
        回归测试：验证最初触发 Bug 的具体条目 José_María_Chacón (sentence_id=0)
        出现在生成的 evidence 文件中。
        """
        evidence_out = generated_files['evidence_out']
        generated = load_generated_evidence(evidence_out)

        target = (unicodedata.normalize('NFC', 'José_María_Chacón'), 0)
        assert target in generated, (
            f'Bug 回归：José_María_Chacón sentence_id=0 仍然缺失于 evidence 输出文件\n'
            f'查找的 NFC page_id repr: {repr(target[0])}\n'
            f'请确认 prepare_fever_llm.process_dataset 中已添加 NFC 规范化修复'
        )

    def test_generated_evidence_json_valid(self, generated_files):
        """验证生成文件的每一行都是合法的 JSON，且包含必要字段。"""
        evidence_out = generated_files['evidence_out']
        invalid_lines = []
        missing_fields_lines = []
        required_fields = {'page_id', 'sentence_id', 'text'}

        with open(evidence_out, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    invalid_lines.append((i, str(e)))
                    continue

                missing = required_fields - set(item.keys())
                if missing:
                    missing_fields_lines.append((i, missing))

        assert not invalid_lines, (
            f'发现 {len(invalid_lines)} 行非法 JSON:\n' +
            '\n'.join(f'  行{i}: {e}' for i, e in invalid_lines[:5])
        )
        assert not missing_fields_lines, (
            f'发现 {len(missing_fields_lines)} 行缺少必要字段:\n' +
            '\n'.join(f'  行{i}: 缺少 {m}' for i, m in missing_fields_lines[:5])
        )
