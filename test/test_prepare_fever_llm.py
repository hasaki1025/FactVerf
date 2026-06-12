"""
test_prepare_fever_llm.py

针对 prepare_fever_llm.process_dataset 的单元测试。
重点检查含特殊 Unicode 字符（如 José_María_Chacón）的 evidence
能否被正确写入输出文件。
"""

import json
import os
import sqlite3
import unicodedata

import pytest

from dataset.FEVER.prepare_fever_llm import build_db, get_evidence_text, process_dataset


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

SPECIAL_PAGE_IDS = [
    # (page_id,            sentence_id, text)
    # NFC 形式（\u00e9 = é 单码点）
    ('José_María_Chacón', 0,
     'Don José María Chacón was the last Spanish Governor of Trinidad.'),
    # NFD 形式（é = e + 组合符号，模拟某些数据源的存储方式）
    (unicodedata.normalize('NFD', 'José_María_Chacón'), 0,
     'NFD variant of the same page.'),
    # 普通 ASCII page_id，作为对照组
    ('Normal_Page', 0, 'This is a normal ASCII page.'),
]


def make_in_memory_db(records):
    """创建内存 SQLite DB 并插入指定记录，返回 connection。"""
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()
    c.execute('CREATE TABLE lines (id TEXT, sentence_id INTEGER, text TEXT)')
    c.execute('CREATE INDEX idx_id_sentence ON lines(id, sentence_id)')
    c.executemany('INSERT INTO lines VALUES (?, ?, ?)', records)
    conn.commit()
    return conn


def make_train_jsonl(records, path):
    """
    生成 train.jsonl 文件。
    records: list of (claim_id, page_id, sentence_id) tuples
    """
    with open(path, 'w', encoding='utf-8') as f:
        for i, (claim_id, page_id, sentence_id) in enumerate(records):
            entry = {
                'id': claim_id,
                'label': 'SUPPORTS',
                'claim': f'Claim {claim_id}',
                'evidence': [[[i, i, page_id, sentence_id]]],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ──────────────────────────────────────────────────────────
# 测试：get_evidence_text 能否通过特殊字符 page_id 查询
# ──────────────────────────────────────────────────────────

class TestGetEvidenceText:
    """验证 get_evidence_text 对各种 Unicode page_id 的查询行为。"""

    def test_ascii_page_id(self):
        """普通 ASCII page_id 应正常返回文本。"""
        conn = make_in_memory_db([('Normal_Page', 0, 'Normal text.')])
        result = get_evidence_text(conn, 'Normal_Page', 0)
        assert result == 'Normal text.'
        conn.close()

    def test_nfc_special_page_id(self):
        """NFC 形式的特殊字符 page_id 应能正常查询。"""
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        conn = make_in_memory_db([(nfc_id, 0, 'Trinidad Governor text.')])
        result = get_evidence_text(conn, nfc_id, 0)
        assert result == 'Trinidad Governor text.', (
            f'NFC page_id 查询失败。'
            f'db 中存储的 id repr: {repr(nfc_id)}'
        )
        conn.close()

    def test_nfd_stored_queried_with_nfc(self):
        """
        DB 中以 NFD 形式存储，用 NFC 查询时的行为。
        SQLite 的 TEXT 比较是字节级别，NFD != NFC，预期返回空字符串。
        此测试用于【文档化】该已知限制，而非断言为正确行为。
        """
        nfd_id = unicodedata.normalize('NFD', 'José_María_Chacón')
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        conn = make_in_memory_db([(nfd_id, 0, 'NFD stored text.')])

        result_with_nfd = get_evidence_text(conn, nfd_id, 0)
        result_with_nfc = get_evidence_text(conn, nfc_id, 0)

        # NFD 查 NFD：应能找到
        assert result_with_nfd == 'NFD stored text.', 'NFD 存储后用 NFD 查询应成功'
        # NFC 查 NFD：SQLite 字节比较，找不到 → 这就是 evidence 丢失的根本原因！
        assert result_with_nfc == '', (
            'NFC 查 NFD 预期返回空（SQLite 字节比较不做 Unicode 规范化）'
        )
        conn.close()

    def test_nfc_stored_queried_with_nfc(self):
        """DB 以 NFC 存储，用 NFC 查询，应成功。"""
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        conn = make_in_memory_db([(nfc_id, 0, 'NFC stored text.')])
        result = get_evidence_text(conn, nfc_id, 0)
        assert result == 'NFC stored text.'
        conn.close()


# ──────────────────────────────────────────────────────────
# 测试：build_db 写入 DB 时的 Unicode 形式
# ──────────────────────────────────────────────────────────

class TestBuildDb:
    """验证 build_db 从 wiki-pages 读取数据写入 DB 时，id 的 Unicode 形式。"""

    def test_wiki_page_nfc_id_stored_correctly(self, tmp_path):
        """
        模拟 wiki-pages jsonl 文件中 id 含特殊字符。
        验证 build_db 写入 DB 后能用相同 id 查询到数据。
        """
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        wiki_content = {
            'id': nfc_id,
            'text': 'Don José María Chacón was the last Spanish Governor of Trinidad.',
            'lines': '0\tDon José María Chacón was the last Spanish Governor of Trinidad.\n1\tHe surrendered to the British in 1797.',
        }
        wiki_file = tmp_path / 'wiki-test.jsonl'
        # ensure_ascii=False：以真实 Unicode 字符写入（不转为 \uXXXX）
        wiki_file.write_text(
            json.dumps(wiki_content, ensure_ascii=False) + '\n',
            encoding='utf-8'
        )

        db_path = str(tmp_path / 'test.db')
        conn = build_db(str(tmp_path), db_path)

        # 用 NFC id 查询
        result = get_evidence_text(conn, nfc_id, 0)
        assert result == 'Don José María Chacón was the last Spanish Governor of Trinidad.', (
            f'build_db 写入后无法用 NFC id 查询到数据。\n'
            f'存储的 id repr: {repr(nfc_id)}'
        )
        conn.close()

    def test_wiki_page_escaped_id_stored_correctly(self, tmp_path):
        """
        模拟 wiki-pages jsonl 文件中 id 以 \\uXXXX 转义形式存储
        （ensure_ascii=True 写入的文件）。
        json.loads 解码后，\\u00e9 变为真正的 é（NFC），build_db 应能写入并查询。
        """
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        wiki_content = {
            'id': nfc_id,
            'text': 'Test',
            'lines': '0\tTest sentence.',
        }
        wiki_file = tmp_path / 'wiki-escaped.jsonl'
        # ensure_ascii=True：以 \u00e9 形式写入（模拟原始 wiki-pages 文件格式）
        wiki_file.write_text(
            json.dumps(wiki_content, ensure_ascii=True) + '\n',
            encoding='utf-8'
        )

        db_path = str(tmp_path / 'test2.db')
        conn = build_db(str(tmp_path), db_path)

        result = get_evidence_text(conn, nfc_id, 0)
        assert result == 'Test sentence.', (
            f'\\uXXXX 转义写入的文件，build_db 解码后应能正确查询。\n'
            f'查询 id repr: {repr(nfc_id)}'
        )
        conn.close()


# ──────────────────────────────────────────────────────────
# 测试：process_dataset 端到端，验证特殊字符 evidence 是否写入输出
# ──────────────────────────────────────────────────────────

class TestProcessDataset:
    """端到端验证 process_dataset 对特殊 Unicode page_id 的处理。"""

    def _run_process(self, tmp_path, train_records, db_records):
        """辅助方法：构造输入文件和 DB，运行 process_dataset，返回输出内容。"""
        input_path = str(tmp_path / 'train.jsonl')
        evidence_out = str(tmp_path / 'evidence_out.jsonl')
        claims_out = str(tmp_path / 'claims_out.jsonl')

        make_train_jsonl(train_records, input_path)
        conn = make_in_memory_db(db_records)

        process_dataset(input_path, evidence_out, claims_out, conn)
        conn.close()

        evidence_items = []
        if os.path.exists(evidence_out):
            with open(evidence_out, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        evidence_items.append(json.loads(line))

        return evidence_items

    def test_ascii_evidence_written(self, tmp_path):
        """ASCII page_id 的 evidence 应正确写入输出文件。"""
        records = [(1001, 'Normal_Page', 0)]
        db_records = [('Normal_Page', 0, 'Normal evidence text.')]

        evidence_items = self._run_process(tmp_path, records, db_records)

        page_ids = [e['page_id'] for e in evidence_items]
        assert 'Normal_Page' in page_ids, '普通 ASCII page_id 应出现在输出中'

    def test_nfc_special_char_evidence_written(self, tmp_path):
        """
        NFC 形式的特殊字符 page_id（José_María_Chacón）应出现在输出文件中。
        这是核心测试：验证端到端流程不会丢失特殊字符 evidence。
        """
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        records = [(132873, nfc_id, 0)]
        db_records = [(nfc_id, 0, 'Don José María Chacón was the last Spanish Governor of Trinidad.')]

        evidence_items = self._run_process(tmp_path, records, db_records)

        page_ids_nfc = [unicodedata.normalize('NFC', e['page_id']) for e in evidence_items]
        assert nfc_id in page_ids_nfc, (
            f'NFC 特殊字符 page_id 应出现在输出文件中，但未找到。\n'
            f'实际输出的 page_ids: {[e["page_id"] for e in evidence_items]}'
        )

    def test_train_jsonl_nfd_page_id_not_found_in_db(self, tmp_path):
        """
        【关键 Bug 复现测试】
        train.jsonl 中的 page_id 为 NFD 形式，DB 中存储的是 NFC 形式。
        SQLite 字节比较导致 get_evidence_text 查不到数据 → evidence 丢失。
        此测试验证 Bug 是否存在，并文档化该已知问题。
        """
        nfd_id = unicodedata.normalize('NFD', 'José_María_Chacón')
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')

        # train.jsonl 中 page_id 为 NFD（模拟原始数据中的 NFD 存储）
        records = [(132873, nfd_id, 0)]
        # DB 中存储 NFC（build_db 从 wiki-pages 读取后 json.loads 解码，通常为 NFC）
        db_records = [(nfc_id, 0, 'Don José María Chacón was the last Spanish Governor of Trinidad.')]

        evidence_items = self._run_process(tmp_path, records, db_records)

        page_ids = [e['page_id'] for e in evidence_items]

        # 若 Bug 存在：evidence_items 为空，因为 NFD 查 NFC 的 DB 时找不到
        # 若 Bug 已修复（process_dataset 中做了规范化）：evidence_items 不为空
        if not evidence_items:
            pytest.fail(
                '【Bug 确认】train.jsonl 中 NFD 形式的 page_id 在 NFC DB 中查不到，'
                'evidence 丢失！\n'
                f'train page_id repr: {repr(nfd_id)}\n'
                f'db    page_id repr: {repr(nfc_id)}\n'
                '修复方案：在 process_dataset 中对 page_id 做 NFC 规范化后再查询 DB。'
            )

    def test_output_file_uses_ensure_ascii_false(self, tmp_path):
        """
        验证输出文件使用 ensure_ascii=False，特殊字符以真实 Unicode 写入，
        而非 \\uXXXX 转义形式。
        """
        nfc_id = unicodedata.normalize('NFC', 'José_María_Chacón')
        records = [(1, nfc_id, 0)]
        db_records = [(nfc_id, 0, 'Trinidad text.')]

        self._run_process(tmp_path, records, db_records)

        evidence_out = str(tmp_path / 'evidence_out.jsonl')
        with open(evidence_out, encoding='utf-8') as f:
            raw = f.read()

        # ensure_ascii=False 时，é 以真实字符写入，不应出现 \u00e9
        assert '\\u00e9' not in raw, (
            '输出文件不应包含 \\u00e9 转义（应使用 ensure_ascii=False 写入真实 Unicode 字符）'
        )
        assert 'é' in raw, '输出文件应包含真实的 é 字符'

    def test_multiple_special_char_evidences(self, tmp_path):
        """验证多个含特殊字符的 evidence 条目均能正确写入。"""
        special_pages = [
            ('José_María_Chacón', 'Text about Chacón.'),
            ('Ångström', 'Text about Ångström unit.'),
            ('Ñoño', 'Text about Ñoño.'),
        ]
        records = [
            (i + 1, unicodedata.normalize('NFC', pid), 0)
            for i, (pid, _) in enumerate(special_pages)
        ]
        db_records = [
            (unicodedata.normalize('NFC', pid), 0, text)
            for pid, text in special_pages
        ]

        evidence_items = self._run_process(tmp_path, records, db_records)

        output_nfc_ids = {unicodedata.normalize('NFC', e['page_id']) for e in evidence_items}
        for pid, _ in special_pages:
            nfc_pid = unicodedata.normalize('NFC', pid)
            assert nfc_pid in output_nfc_ids, (
                f'特殊字符 page_id {repr(nfc_pid)} 未出现在输出中'
            )
