import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location('publication', Path(__file__).resolve().parents[1] / 'scripts/v31_publish_research_week.py')
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


@pytest.mark.parametrize('field', ['prompt', 'completion', 'gold', 'token_ids', 'output_token_ids', 'api_key'])
def test_nested_raw_payload_is_rejected(field):
    with pytest.raises(ValueError, match='refusing'):
        publication.sanitize({'arms': [{'nested': {field: 'private'}}]})


def test_source_hashes_and_counts_survive_path_redaction():
    value = {'source': '/home/researcher/private/model', 'output_tokens': 42,
             'answer_text': {'file': 'experiments/protocol.py', 'sha256': 'abc'}}
    result = publication.sanitize(value)
    assert result['source'].startswith('<private_path:')
    assert result['output_tokens'] == 42
    assert result['answer_text']['sha256'] == 'abc'


def test_historical_early_columns_are_excluded_without_changing_total_work():
    data = b'threshold,global_kept_tiles,early_kept_tiles,first_C_cumulative_kept_tiles\r\n0.021,123,999,888\r\n'
    assert publication.export_csv(data, total_curve=True) == b'threshold,global_kept_tiles\n0.021,123\n'


def test_csv_payload_columns_are_rejected():
    with pytest.raises(ValueError, match='raw field'):
        publication.export_csv(b'threshold,completion\n0.021,private\n')


def test_existing_campaign_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    target = tmp_path / 'results' / 'existing'
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match='new output'):
        publication.export(target)


@pytest.mark.parametrize('script, flags', [
    ('v31_value_v2_equal_work.py', ['--model', '--manifests', '--study', '--development-cells']),
    ('v31_build_quick_accuracy_suite.py', ['--tokenizer', '--math500', '--gsm8k', '--longbench-zip']),
])
def test_portable_driver_help_requires_no_private_input(script, flags):
    result = subprocess.run([sys.executable, str(publication.ROOT / 'scripts' / script), '--help'],
                            capture_output=True, text=True, check=True)
    assert all(flag in result.stdout for flag in flags)
