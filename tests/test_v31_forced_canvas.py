"""v31 forced-canvas mode of the paired bench (CanvasForcing): per-canvas step counts, per-canvas reseeding before every
commit, reference overwrite of a converged canvas, agreement (CPU; a fake sampler drives the state machine)."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parents[1] / 'scripts' / 'v31_vllm_paired_bench.py'
if not BENCH.exists():                                    # deployed overlay layout: tests next to the bench
    BENCH = Path(__file__).resolve().parent / 'v31_vllm_paired_bench.py'


def _bench():
    spec = importlib.util.spec_from_file_location('_v31_bench_under_test', BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeSampler:
    """Mimics _compiled_sample_step's state machine for one slot: a denoise call converges after the scripted number
    of steps (argmax = the scripted tokens, is_encoder_phase -> True); the next call commits (-> False)."""

    def __init__(self, steps, tokens, CL):
        self.steps, self.tokens, self.CL, self.i, self.n = list(steps), list(tokens), CL, 0, 0
        self.emitted = []

    def __call__(self, *args, **kwargs):
        slots, canvas, argmax, enc, draft = args[1], args[5], args[6], args[8], args[17]
        s = int(slots[0])
        if bool(enc[s]):                                   # commit: emit the (possibly forced) argmax canvas
            self.emitted.append(argmax[s].clone())
            enc[s] = False
            canvas[s] = -1
            return 'commit'
        self.n += 1
        if self.n == self.steps[self.i]:
            argmax[s] = torch.tensor(self.tokens[self.i])
            canvas[s] = argmax[s]
            draft[s, :self.CL] = canvas[s]
            enc[s] = True
            self.i, self.n = self.i + 1, 0
        return 'denoise'


def _args(CL):
    canvas, argmax = torch.zeros(1, CL, dtype=torch.long), torch.zeros(1, CL, dtype=torch.long)
    enc, draft = torch.zeros(1, dtype=torch.bool), torch.zeros(1, CL + 2, dtype=torch.int32)
    args = [None, torch.tensor([0]), None, None, None, canvas, argmax, None, enc] + [None] * 8 + [draft]
    return args, canvas, argmax, enc, draft


def test_forcing_counts_steps_reseeds_and_overwrites(tmp_path=None, monkeypatch=None):
    mod = _bench()
    CL = 8
    ref = list(range(100, 120))                            # 3 canvases: 8 + 8 + 4 tokens
    arm_tokens = [list(range(100, 108)),                   # canvas 0: the arm agrees fully
                  [108, 109, 0, 0, 112, 113, 114, 115],    # canvas 1: 6 / 8 agree
                  [116, 117, 118, 1, 9, 9, 9, 9]]          # canvas 2: 3 / 4 of the reference's real tokens agree
    path = Path(__file__).resolve().parent / '_forced_ref_test.jsonl'
    path.write_text(json.dumps(dict(dataset='d', index=0, panel_seed=3, repeat=0, token_ids=ref)) + '\n')
    try:
        seeds = []
        real = torch.cuda.manual_seed
        torch.cuda.manual_seed = lambda x: seeds.append(x)
        try:
            f = mod.CanvasForcing(str(path), None)
            fake = FakeSampler([5, 3, 7], arm_tokens, CL)
            step = f.wrap(fake)
            f.begin(1234, ('d', 0, 3, 0))
            args, canvas, argmax, enc, draft = _args(CL)
            for _ in range(5 + 1 + 3 + 1 + 7 + 1):
                step(*args, CL=CL)
        finally:
            torch.cuda.manual_seed = real
        assert f.steps_list == [5, 3, 7]
        assert f.agree == [1.0, 0.75, 0.75]
        assert seeds == [mod.canvas_seed(1234, 1), mod.canvas_seed(1234, 2), mod.canvas_seed(1234, 3)]
        out = torch.cat([fake.emitted[0], fake.emitted[1], fake.emitted[2][:4]]).tolist()
        assert out == ref                                  # every commit emitted the reference tokens
        assert int(draft[0, 2]) == 118 and int(draft[0, 5]) == 9   # the reference prefix, the arm's tail kept
        r = f.receipt(out)
        assert r['forced_mode'] == 'force' and r['forced_output_matches_ref'] and r['forced_canvas_steps'] == [5, 3, 7]
    finally:
        path.unlink()


def test_record_mode_only_reseeds_and_counts():
    mod = _bench()
    CL = 8
    f = mod.CanvasForcing(None, 'unused')
    fake = FakeSampler([2, 4], [list(range(8)), list(range(8, 16))], CL)
    step = f.wrap(fake)
    seeds = []
    real = torch.cuda.manual_seed
    torch.cuda.manual_seed = lambda x: seeds.append(x)
    try:
        f.begin(7, ('d', 0, 3, 0))
        args, *_ = _args(CL)
        for _ in range(2 + 1 + 4 + 1):
            step(*args, CL=CL)
    finally:
        torch.cuda.manual_seed = real
    assert f.mode == 'record' and f.steps_list == [2, 4] and f.agree == [] and len(seeds) == 2
    assert torch.cat(fake.emitted).tolist() == list(range(16))   # nothing overwritten
    assert 'forced_output_matches_ref' not in f.receipt(list(range(16)))


def test_missing_reference_is_refused():
    mod = _bench()
    path = Path(__file__).resolve().parent / '_forced_ref_test2.jsonl'
    path.write_text(json.dumps(dict(dataset='d', index=0, panel_seed=3, repeat=0, token_ids=[1, 2])) + '\n')
    try:
        f = mod.CanvasForcing(str(path), None)
        try:
            f.begin(1, ('d', 1, 3, 0))
            raise AssertionError('missing reference accepted')
        except KeyError:
            pass
    finally:
        path.unlink()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    sys.exit(0)
