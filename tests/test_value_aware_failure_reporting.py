"""Error disclosure must distinguish attempt history from final completeness."""
import json
from experiments.diffusion_gemma_value_aware.scientific_report import execution_failure_rows


def test_absent_error_log(tmp_path):
    assert execution_failure_rows(tmp_path)==[]


def test_preserved_attempts_group_by_stage_and_exception_not_final_status(tmp_path):
    oom='Traceback...\ntorch.OutOfMemoryError: CUDA out of memory. Process 340198 has 28.94 GiB memory in use.'
    entries=[dict(stage='calibration',condition='mass_s25/aime26/point',id='aime26/2',traceback=oom),
        dict(stage='calibration',condition='mass_s25/aime26/point',id='aime26/2',traceback=oom),
        dict(stage='calibration',condition='mass_s25/aime26/point',id='aime26/8',traceback=oom),
        dict(stage='final',condition='mass_s25',id='aime26/8',traceback='Traceback...\nValueError: invalid cache'),
        dict(stage='development',condition='mass_s25',error='missing verified policy')]
    (tmp_path/'failures.jsonl').write_text('\n'.join(json.dumps(r) for r in entries)+'\n\n')
    rows=execution_failure_rows(tmp_path)
    assert rows[0]==dict(stage='calibration',condition='mass_s25/aime26/point',
        error_type='torch.OutOfMemoryError',failed_attempts=3,distinct_sample_ids=2,
        reported_other_gpu_pids=['340198'])
    assert rows[1]['stage']=='development' and rows[1]['distinct_sample_ids']==0
    assert rows[2]['stage']=='final' and rows[2]['error_type']=='ValueError'
    assert sum(r['failed_attempts'] for r in rows)==5
    assert all('recovered' not in r and 'missing_final' not in r for r in rows)
