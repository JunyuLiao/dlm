"""v27 dataset names and their base task contracts (one place for freeze, run and score).

A long-context set runs its base task's contract unchanged (prompt rendering, thinking,
budget and scorer); only its manifest and gold differ, and its gold is pinned by sha256
in the frozen protocol (``extra_gold_sha256``) instead of the v20 source identity.
"""
BASE_TASK = {
    'longbench_v2': 'longbench_v2', 'aime26': 'aime26', 'ruler4k': 'ruler4k',
    'ruler32k': 'ruler4k', 'ruler64k': 'ruler4k',
    'longbench_v2_32k': 'longbench_v2', 'longbench_v2_64k': 'longbench_v2', 'longbench_v2_128k': 'longbench_v2',
    'longbench_v2_96k': 'longbench_v2',
    'humaneval': 'humaneval',
}
DATASETS = tuple(BASE_TASK)
EXTRA_GOLD = tuple(d for d, base in BASE_TASK.items() if d != base or d == 'humaneval')
# v27 generation seeds: 101/202/303 panels, 404 final AIME panel, 505-909 the E4 large-seed confirmation,
# 1010-1515 the E14 fresh-seed confirmation, 1616-2121 the E15 observe_carried panel (each never used before)
V27_SEEDS = frozenset({101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515,
                       1616, 1717, 1818, 1919, 2020, 2121})


def base_task(dataset: str) -> str:
    try:
        return BASE_TASK[dataset]
    except KeyError:
        raise ValueError(f'unknown dataset {dataset!r}') from None


def runtime_base_task(dataset: str) -> str:
    """Reuse only the pinned thinking-ON/8192 runtime config, never the LB scorer."""
    base = base_task(dataset)
    return 'longbench_v2' if base == 'humaneval' else base
