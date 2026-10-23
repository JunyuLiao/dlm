import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1]/'code/candidate_readout.py'
spec = importlib.util.spec_from_file_location('candidate_readout_test', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def row(label, support, response, error, future, feasible=True, fraction=.5):
    return dict(request='request002', phase='early', head=0, kept_prefix_blocks=2,
        support_sha256=support, response_gain={'total': response}, output_bias_relative_max=error,
        reuse=[dict(bias_motion_relative_max=future, output_bias_relative_max=future)],
        candidate=label, split='heldout', dataset='toy', response_guard_feasible=feasible,
        budget_fraction=fraction)

def test_oracle_uses_only_current_response_and_deduplicates_physical_support():
    mass = row('mass', 'a', 2., .1, .01)
    lower_response = row('swap', 'b', 1., .2, .3)
    duplicate = dict(mass, budget_fraction=.75)
    result = module.analyze([mass, lower_response, duplicate])
    assert result['unique_physical_records'] == 2
    assert result['physical_groups'] == 1
    group = result['groups'][0]
    assert group['response']['candidate'] == 'swap'
    assert group['error']['candidate'] == 'mass'
    assert result['summary']['heldout/toy']['versus_error']['higher'] == 1

def test_no_feasible_candidate_records_mass_fallback():
    result = module.analyze([row('mass', 'a', 2., .1, .01, False),
                             row('swap', 'b', 1., .2, .3, False)])
    assert result['groups'][0]['fallback']
    assert result['groups'][0]['response']['candidate'] == 'mass'
