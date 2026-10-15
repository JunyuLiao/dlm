import pytest
from scripts.v31_value_process_matched_study import CONTROL,expand,validate_group,write_or_verify


def row(n=20,active=False):
    return dict(output_hash=str(n),denoise_forwards=n,canvases=2,output_tokens=400,rng_seed=42,
                receipts=dict(adapter=dict(value_refinement_work=dict(corrected_calls=int(active)))))


def test_latin_order_pairs_exact_cells_and_seeds_and_control_bookends():
    modes=['full_swap8','residual','residual_direction_guard','full_swap8_direction_guard']
    cells=[dict(index=i,seed=42) for i in range(4)]
    rows,arms,groups=expand(cells,modes)
    assert len(rows)==24 and len(arms)==25 and arms[0]==CONTROL
    seen=[]
    for cell,g in zip(cells,groups):
        assert rows[g['start']:g['end']]==[cell]*6
        assert g['arms'][0]==g['arms'][-1]==CONTROL
        assert sorted(g['arms'][1:-1])==sorted(modes)
        assert arms[g['start']+1:g['end']+1]==g['arms']
        seen.append(g['arms'][1])
    assert set(seen)==set(modes)


def test_active_change_is_permitted_but_control_or_inactive_drift_is_rejected():
    arms=[CONTROL,'residual',CONTROL]
    assert validate_group([row(),row(25,True),row()],arms)==[]
    assert validate_group([row(),row(25),row()],arms)==['inactive candidate drift: residual']
    assert 'control bookend drift' in validate_group([row(),row(25,True),row(21)],arms)


def test_resume_preserves_identical_report_and_rejects_changed_result(tmp_path):
    path=tmp_path/'summary.json'
    write_or_verify(path,dict(score=1),True)
    original=path.read_bytes()
    write_or_verify(path,dict(score=1),True)
    assert path.read_bytes()==original
    with pytest.raises(RuntimeError):write_or_verify(path,dict(score=2),True)
    assert path.read_bytes()==original
