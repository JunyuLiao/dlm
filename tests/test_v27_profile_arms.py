"""v27 direct-cost arm table: every named point builds a valid v21 identity."""
import copy

from experiments.numerical_qk_reuse import v21
from scripts.v24_profile import ARM_SETS, BOOT, SHIFTS, V27

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def test_every_v27_point_builds_and_validates():
    for name, (parent, extra) in V27.items():
        extra = dict(extra)
        scope = extra.pop('scope', 'GLOBAL_ONLY_NATIVE_LOCAL')
        config = v21.effective_config(copy.deepcopy(BASE), parent, scope,
                                      output_score_precision='fp32_scores_bf16_pv', output_layout='model_major',
                                      bootstrap_policy=BOOT, observation_producer='grouped_q',
                                      route_storage='aligned16_odd', **extra)
        v21.validate_effective(config, config['condition'])


def test_arm_sets_start_with_native_and_have_unique_names():
    for name in ('v27', 'v27thr'):
        arms = ARM_SETS[name]
        assert arms[0] == 'D_native' and len(set(arms)) == len(arms)
    assert ARM_SETS['v27thr'][1:] == tuple('M3_R3_A8_' + ('P0' if s is None else s) for s in SHIFTS)
    for name in ARM_SETS:
        assert len(set(ARM_SETS[name])) == len(ARM_SETS[name]) and all(n in V27 or n.startswith(('D_', 'T_scope')) or n in ('M3_R3_A8_incumbent', 'M1_native_bootstrap2_observe1', 'M3_native_bootstrap2_observe1', 'B_native_bootstrap2_observe1', 'M3_boot_aligned16', 'B_boot_aligned16') for n in ARM_SETS[name])
    assert V27['B_A16'][1] == dict(hold_only=True, score_period=16)
