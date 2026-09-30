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
}
DATASETS = tuple(BASE_TASK)
EXTRA_GOLD = tuple(d for d, base in BASE_TASK.items() if d != base)


def base_task(dataset: str) -> str:
    try:
        return BASE_TASK[dataset]
    except KeyError:
        raise ValueError(f'unknown dataset {dataset!r}') from None
