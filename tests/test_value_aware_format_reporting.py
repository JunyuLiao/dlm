"""Report inherited output-format failures without changing frozen scores."""
import json

from experiments.diffusion_gemma_value_aware.protocol import sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.scientific_report import (
    dense_trec_format_rows, trec_format_note,
)


def test_format_audit_uses_only_final_dense_trec_shards(tmp_path):
    row = dict(id='longbench/trec/test', benchmark='longbench', task='trec')
    path = shard_path(tmp_path, 'final', 'dense', row['id'])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(prediction='\nthought\nAnimal', score=0.)))
    before = path.read_bytes()
    rows = dense_trec_format_rows(tmp_path, dict(final=[row,
        dict(id='unread', benchmark='aime26', task='AIME26')]))
    assert len(rows) == 1
    assert rows[0]['first_line'] == 'thought'
    assert rows[0]['first_line_is_thought']
    assert rows[0]['stored_score'] == 0.
    assert rows[0]['source_sha256'] == sha(before)
    assert path.read_bytes() == before


def test_format_audit_does_not_strip_extra_whitespace(tmp_path):
    row = dict(id='trec/space', benchmark='longbench', task='trec')
    path = shard_path(tmp_path, 'final', 'dense', row['id'])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(prediction=' thought\nAnimal', score=0.)))
    result = dense_trec_format_rows(tmp_path, dict(final=[row]))[0]
    assert result['first_line'] == ' thought'
    assert not result['first_line_is_thought']


def test_format_note_reports_counts_and_preserves_protocol():
    note = trec_format_note([dict(first_line_is_thought=True)] * 10)
    assert '10/10 dense' in note
    assert 'Canonical scores, prompts and output parsing remain unchanged' in note
    assert 'dense_trec_format_audit.json' in note


def test_no_format_warning_without_observed_prelude():
    assert trec_format_note([]) == ''
    assert trec_format_note([dict(first_line_is_thought=False)]) == ''
