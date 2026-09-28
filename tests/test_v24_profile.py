"""CPU checks for v24 phase classification (no model, no GPU)."""
from scripts.v24_profile import ARMS, classify


def test_five_phases_and_native_are_disjoint():
    assert classify({}) == 'native'
    assert classify({'bootstrap_dense_calls': 5}) == 'B0'
    assert classify({'bootstrap_observation_calls': 5}) == 'BO'
    assert classify({'score_refresh_calls': 5, 'decision_refresh_calls': 5, 'attention_calls': 5}) == 'A'
    assert classify({'decision_refresh_calls': 5, 'attention_calls': 5}) == 'D'
    assert classify({'held_decision_calls': 5, 'attention_calls': 5}) == 'H'
    assert classify({'calls': 5}) == 'routed_other'
    assert classify({'score_refresh_calls': 0, 'held_decision_calls': 0}) == 'native'


def test_arm_inventory_is_the_frozen_panel():
    assert ARMS == ('D_native', 'T_scope', 'M3_R3_A8_incumbent', 'M1_native_bootstrap2_observe1',
                    'M3_native_bootstrap2_observe1', 'B_native_bootstrap2_observe1')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('OK', name)
