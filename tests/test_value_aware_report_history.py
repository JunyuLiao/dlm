"""A cache-linked sub-bundle must not lose prior negative evidence."""
import json
from pathlib import Path

import pytest

from experiments.diffusion_gemma_value_aware.protocol import sha
from experiments.diffusion_gemma_value_aware.scientific_report import (
    historical_artifact_links, historical_evidence_root, report_failure_rows,
    reproduction_command,
)


def setup_history(tmp_path):
    root = tmp_path/'reduced'
    root.mkdir()
    source = tmp_path/'final_contract.json'
    source.write_text('{"conditions": {}}')
    digest = sha(source.read_bytes())
    plan = dict(original_contract=str(source), original_contract_sha256=digest)
    (root/'iteration_plan.json').write_text(json.dumps(plan))
    contract = dict(user_authorized_scope_reduction=True,
        original_contract_sha256=digest, sources={str(source): digest})
    return root, source, contract


def test_unreduced_report_keeps_own_history(tmp_path):
    assert historical_evidence_root(tmp_path, {}) == tmp_path


def test_reduced_report_resolves_only_pinned_parent(tmp_path):
    root, source, contract = setup_history(tmp_path)
    assert historical_evidence_root(root, contract) == tmp_path
    source.write_text('{}')
    with pytest.raises(ValueError, match='not pinned'):
        historical_evidence_root(root, contract)


def test_parent_contract_must_be_in_frozen_sources(tmp_path):
    root, _, contract = setup_history(tmp_path)
    contract['sources'] = {}
    with pytest.raises(ValueError, match='not pinned'):
        historical_evidence_root(root, contract)


def test_parent_and_current_failure_attempts_remain_distinct(tmp_path):
    root, _, _ = setup_history(tmp_path)
    entry = dict(stage='calibration', condition='mass_s50', id='cal1',
        traceback='ValueError: test error')
    for directory in (tmp_path, root):
        (directory/'failures.jsonl').write_text(json.dumps(entry)+'\n')
    rows = report_failure_rows(root, tmp_path)
    assert len(rows) == 2 and sum(r['failed_attempts'] for r in rows) == 2
    assert {Path(r['bundle']) for r in rows} == {tmp_path, root}
    assert len(report_failure_rows(tmp_path, tmp_path)) == 1


def test_history_links_point_to_existing_parent_artifacts_only(tmp_path):
    root, _, _ = setup_history(tmp_path)
    (tmp_path/'execution_notes.md').write_text('Historical notes')
    links = historical_artifact_links(root, tmp_path)
    assert links == '[execution_notes.md](../execution_notes.md)'
    assert 'screening_pruning' not in links


def test_reduced_reproduction_uses_reduced_entrypoint_and_explicit_parent(tmp_path):
    import shlex
    history = tmp_path/'experiment with spaces'
    root = history/'efficient_iteration'
    args = shlex.split(reproduction_command(root, history))
    assert args[:2] == ['CUDA_VISIBLE_DEVICES=', 'PYTHONPATH=src:.']
    assert args[-4:] == ['experiments.diffusion_gemma_value_aware.efficient_iteration',
        'report', '--root', str(history)]
    assert 'experiments.diffusion_gemma_value_aware.report' not in args


def test_original_reproduction_names_its_actual_output_directory(tmp_path):
    import shlex
    args = shlex.split(reproduction_command(tmp_path, tmp_path))
    assert args[-3:] == ['experiments.diffusion_gemma_value_aware.report',
        '--output', str(tmp_path)]
