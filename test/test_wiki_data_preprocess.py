from dataset.FEVER.prepare_fever_llm import build_db, get_evidence_text


def test_get_wiki_db():
    conn = build_db('dataset/FEVER/wiki-pages', 'test/wiki.db')
    assert get_evidence_text(conn,'José María Chacón', 0 ) != ''