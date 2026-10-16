from scripts.v31_value_native_noop import repeated_outputs_identical


def row(index,n,seed=31):
    return dict(dataset='diagnostic',index=index,panel_seed=42,repeat=0,rng_seed=seed,
                output_hash=str(n),denoise_forwards=n)


def test_repeated_history_groups_do_not_compare_different_problems():
    assert repeated_outputs_identical([row(0,20),row(0,20),row(10,400),row(10,400)],
                                      ('output_hash','denoise_forwards','rng_seed'))


def test_late_drift_or_rng_change_in_one_repeated_group_is_detected():
    assert not repeated_outputs_identical([row(0,20),row(0,20),row(10,400),row(10,420)],
                                          ('output_hash','denoise_forwards','rng_seed'))
    assert not repeated_outputs_identical([row(10,400),row(10,400,seed=32)],
                                          ('output_hash','denoise_forwards','rng_seed'))
