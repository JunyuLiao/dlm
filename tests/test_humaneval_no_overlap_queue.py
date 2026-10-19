from pathlib import Path

import pytest

from scripts import v31_humaneval_no_overlap_queue as queue


def complete_panel(root):
    queue.write(root / 'protocol.json', {'scope': 'test'}, True)
    analysis = root / 'analysis/attempt001'
    analysis.mkdir(parents=True)
    queue.write(analysis / 'summary.json', {'status': 'complete'}, True)
    queue.write(analysis / 'analysis_complete.json', {
        'files_sha256': {'summary.json': queue.sha(analysis / 'summary.json')}}, True)
    digest = queue.sha(root / 'protocol.json')
    queue.write(root / 'final_complete.json', {
        'status': 'complete', 'protocol_sha256': digest, 'official_rescore_mismatches': 0,
        'analysis_complete_sha256': queue.sha(analysis / 'analysis_complete.json')}, True)
    return digest


def test_only_complete_bound_panel_advances(tmp_path):
    digest = complete_panel(tmp_path)
    queue.require_complete(tmp_path, digest)
    with pytest.raises(RuntimeError, match='protocol changed'):
        queue.require_complete(tmp_path, 'different')


def test_changed_completed_analysis_is_rejected(tmp_path):
    digest = complete_panel(tmp_path)
    (tmp_path / 'analysis/attempt001/summary.json').write_text('{}')
    with pytest.raises(RuntimeError, match='analysis artifact changed'):
        queue.require_complete(tmp_path, digest)


def test_absent_completion_cannot_advance(tmp_path):
    queue.write(tmp_path / 'protocol.json', {}, True)
    with pytest.raises(FileNotFoundError):
        queue.require_complete(tmp_path, queue.sha(tmp_path / 'protocol.json'))


def test_pid_reuse_zombie_and_absent_process(monkeypatch):
    # starttime is field22; the command name can itself contain parentheses.
    stat = '123 (worker (test)) S ' + ' '.join(['0'] * 18 + ['987654', '0'])
    monkeypatch.setattr(Path, 'read_text', lambda self: stat)
    assert queue.process_identity(123) == '987654'
    monkeypatch.setattr(Path, 'read_text', lambda self: stat.replace(' S ', ' Z '))
    assert queue.process_identity(123) is None
    def absent(self):
        raise FileNotFoundError
    monkeypatch.setattr(Path, 'read_text', absent)
    assert queue.process_identity(123) is None
