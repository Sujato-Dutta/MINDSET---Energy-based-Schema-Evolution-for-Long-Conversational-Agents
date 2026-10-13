import json
import pytest
from mindset.acl.io import append_row, read_rows, unique_rows


def test_jsonl_preserves_unicode_separators_inside_source_strings(tmp_path):
    p = tmp_path / 'rows.jsonl'
    rows = [{'id': 'one', 'text': 'A\u2028B\u2029C\u0085D\nE'}, {'id': 'two', 'text': 'ordinary'}]
    for row in rows: append_row(p, row)
    assert read_rows(p) == rows
    assert unique_rows(p) == {r['id']: r for r in rows}


def test_jsonl_duplicate_ids_are_still_rejected(tmp_path):
    p = tmp_path / 'rows.jsonl'
    append_row(p, {'id': 'same', 'text': 'A\u2028B'})
    append_row(p, {'id': 'same', 'text': 'different'})
    with pytest.raises(ValueError, match='Duplicate'): unique_rows(p)


def test_jsonl_truncated_record_is_not_silently_skipped(tmp_path):
    p = tmp_path / 'rows.jsonl'
    p.write_text('{"id": "broken", "text": "unfinished', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError): read_rows(p)
