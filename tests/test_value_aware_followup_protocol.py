import copy

import pytest

from experiments.diffusion_gemma_value_aware_followup.protocol import (
    DOMAINS, choose_samples, official_extractor, official_prompt, score,
)


def pool():
    return [dict(_id=f'{domain}/{difficulty}/{i}', domain=domain,
        difficulty=difficulty, length='short' if i < 6 else 'medium',
        answer='A', context='unused') for domain in DOMAINS
        for difficulty in ('easy', 'hard') for i in range(9)]


def test_deterministic_difficulty_balanced_disjoint_split():
    first=choose_samples(pool());second=choose_samples(list(reversed(pool())))
    assert first==second
    assert {k:len(v) for k,v in first.items()} == dict(calibration=6,final=30,development=6)
    identities=[{r['_id'] for r in first[k]} for k in ('calibration','final','development')]
    assert not identities[0]&identities[1]
    assert not identities[0]&identities[2]
    assert not identities[1]&identities[2]
    assert all(r['length']=='short' for r in first['final']+first['calibration'])
    assert all(r['length']=='medium' for r in first['development'])
    for domain in DOMAINS:
        for difficulty in ('easy','hard'):
            assert sum(r['domain']==domain and r['difficulty']==difficulty for r in first['final'])==5


def test_selection_never_depends_on_answer():
    a=pool();b=copy.deepcopy(a)
    for row in b:row['answer']='D'
    assert {k:[r['_id'] for r in v] for k,v in choose_samples(a).items()} == {
        k:[r['_id'] for r in v] for k,v in choose_samples(b).items()}


def test_insufficient_or_duplicate_source_pool_rejected():
    with pytest.raises(ValueError,match='insufficient'):
        choose_samples(pool()[:3])
    rows=pool();rows.append(rows[0])
    with pytest.raises(ValueError,match='duplicate'):
        choose_samples(rows)


@pytest.mark.parametrize('text,answer',[
    ('thought\nThe correct answer is (C)', 'C'),
    ('The **correct answer** is B', 'B'),
    ('The correct answer is (A)\nThe correct answer is (D)', 'A'),
    ('C', None),
    ('The correct answer is (E)', None),
    ('the correct answer is (a)', None),
])
def test_exact_official_v2_extractor(text,answer):
    assert official_extractor()(text)==answer


def test_official_metric_does_not_use_first_line_or_guess_letters():
    row=dict(benchmark='longbench_v2',expected='D')
    assert score(row,'thought\nThe correct answer is (D)')==1.
    assert score(row,'D')==0.
    with pytest.raises(ValueError,match='unexpected benchmark'):
        score(dict(benchmark='longbench',expected='D'),'D')


def test_official_prompt_substitution():
    row=dict(context=' text ',question=' query ',choice_A=' one ',choice_B='two',choice_C='three',choice_D='four')
    assert official_prompt(row,'$DOC$|$Q$|$C_A$|$C_B$|$C_C$|$C_D$')=='text|query|one|two|three|four'
