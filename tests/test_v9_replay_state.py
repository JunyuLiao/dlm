"""v9 section 3 regression tests: dispatch selection, per-repetition replay
state, capture-wrapper leakage, real ScoreCache phases, telemetry growth.

Each test first demonstrates the v8 defect on the same fixture where that is
possible, so a passing suite cannot be explained by a fixture that never
exercised the bug.
"""
from types import SimpleNamespace

import pytest
import torch

from experiments.numerical_qk_reuse.cache import Identity, ScoreCache
from scripts.replay_harness import (ObjectState, StepSnapshot, output_digest, reset_router,
                                    telemetry, tensor_digest, unwrap_sampler)


# ---------------------------------------------------------------- toy native step
class ToySampler:
    """Mimics EntropyBoundSampler: a mutable mask written by accept, read by renoise."""

    def __init__(self):
        self.vocab_size, self.accepted_token_mask = 50, None

    def accept_canvas(self, current, proposed, logits, cur_step):
        self.accepted_token_mask = logits.argmax(-1) % 2 == 0
        return torch.where(self.accepted_token_mask, proposed, current)

    def renoise_canvas(self, accepted, cur_step):
        noise = torch.randint(0, self.vocab_size, accepted.shape)
        return torch.where(~self.accepted_token_mask, noise, accepted)


class ToyStopping:
    """Mimics StableAndConfidentStoppingCriteria: history REPLACED every call."""

    def __init__(self):
        self.argmax_canvas_history = None

    def __call__(self, argmax, logits):
        if self.argmax_canvas_history is None:
            self.argmax_canvas_history = torch.full((1,) + argmax.shape, -1)
        stable = (self.argmax_canvas_history == argmax[None]).all()
        self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, -1, 0)
        self.argmax_canvas_history[-1] = argmax
        return stable.reshape(1)


def toy_step(*, current_canvas, logits, finished, sampler, diffusion_stopping_criteria, cur_step):
    proposed = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
    accepted = sampler.accept_canvas(current_canvas, proposed, logits, cur_step)
    new = sampler.renoise_canvas(accepted, cur_step)
    finished |= diffusion_stopping_criteria(proposed, logits)   # in-place, like native
    return new, proposed, finished


def toy_kwargs(sampler=None):
    generator = torch.Generator().manual_seed(3)
    return dict(current_canvas=torch.randint(0, 50, (16,), generator=generator),
                logits=torch.randn(16, 50, generator=generator), finished=torch.zeros(1, dtype=torch.bool),
                sampler=sampler or ToySampler(), diffusion_stopping_criteria=ToyStopping(), cur_step=40)


def test_v8_style_single_restore_lets_draws_and_objects_drift():
    """The defect: one RNG restore around the series + tensor-only clones."""
    torch.manual_seed(0)
    base = toy_kwargs()
    rng = torch.get_rng_state()
    outputs = []
    for _ in range(3):
        kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        outputs.append(output_digest(toy_step(**kwargs)))
    torch.set_rng_state(rng)
    assert len(set(outputs)) > 1                                     # draws advanced
    assert base['diffusion_stopping_criteria'].argmax_canvas_history is not None   # object mutated


def test_per_repetition_restore_gives_identical_inputs_state_and_outputs():
    torch.manual_seed(0)
    base = toy_kwargs()
    snapshot = StepSnapshot(base)
    inputs, outputs = [], []
    for _ in range(4):
        kwargs = snapshot.prepare()
        inputs.append(StepSnapshot.digest(kwargs))
        outputs.append(output_digest(toy_step(**kwargs)))
    assert len(set(inputs)) == 1 and len(set(outputs)) == 1
    # A restore (before the next repetition) returns the objects to their
    # pre-replay state, however the last replay left them.
    snapshot.prepare()
    assert base['diffusion_stopping_criteria'].argmax_canvas_history is None
    assert base['sampler'].accepted_token_mask is None


def test_restored_sampler_tensor_is_not_aliased_to_the_saved_copy():
    sampler = ToySampler()
    sampler.accepted_token_mask = torch.zeros(4, dtype=torch.bool)
    state = ObjectState(sampler)
    sampler.accepted_token_mask[0] = True             # in-place mutation
    sampler.extra = 1                                  # attribute created by a replay
    state.restore()
    assert not sampler.accepted_token_mask.any() and not hasattr(sampler, 'extra')
    sampler.accepted_token_mask[1] = True
    state.restore()
    assert not sampler.accepted_token_mask.any()


def test_capture_wrapper_is_unwrapped_and_its_state_never_observes_replays():
    from experiments.value_direction_hopper.query_adaptive import Sampler, State
    capture_state = State('T', None, m_ref=1., diagnostics=False)
    capture_state.current = dict(remaining_schedule_step=40)
    calls = []
    capture_state.observe_logits = lambda *a, **k: calls.append(1)
    inner = ToySampler()
    wrapped = Sampler(Sampler(inner, capture_state), capture_state)
    assert unwrap_sampler(wrapped) is inner
    torch.manual_seed(1)
    leaky = toy_kwargs(sampler=wrapped)
    toy_step(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in leaky.items()})
    assert calls, 'fixture must show the v8 leak'     # v8: capture State kept observing
    calls.clear()
    snapshot = StepSnapshot(toy_kwargs(sampler=wrapped))
    for _ in range(2):
        kwargs = snapshot.prepare()
        assert type(kwargs['sampler']) is ToySampler
        toy_step(**kwargs)
    assert not calls


def test_controller_state_is_restored_before_each_repetition():
    from experiments.value_direction_hopper.query_adaptive import State
    captured = State('T', None, m_ref=1., diagnostics=False)
    captured.begin(48, torch.zeros(1, 8, dtype=torch.long))
    captured.observe_logits(torch.randn(1, 8, 20), torch.ones(1, 8, dtype=torch.bool), 48)
    snapshot = StepSnapshot(toy_kwargs(), controller=captured)
    live = State('T', SimpleNamespace(query_sensitivity=None), m_ref=1., diagnostics=False)
    router = live.router
    digests = []
    for _ in range(3):
        snapshot.prepare(controller=live)
        assert live.router is router                    # production router is kept
        digests.append(StepSnapshot.digest({}, controller=live))
        live.begin(47, torch.zeros(1, 8, dtype=torch.long))         # advances T history
        live.observe_logits(torch.randn(1, 8, 20), torch.ones(1, 8, dtype=torch.bool), 47)
    assert len(set(digests)) == 1
    assert live.iteration == captured.iteration + 1


def test_bit_level_digest_sees_signed_zero_and_order():
    assert tensor_digest(torch.tensor([0.])) != tensor_digest(torch.tensor([-0.]))
    assert tensor_digest(torch.tensor([1., 2.])) != tensor_digest(torch.tensor([2., 1.]))
    assert tensor_digest(torch.tensor([1., 2.], dtype=torch.bfloat16)) == \
        tensor_digest(torch.tensor([1., 2.], dtype=torch.bfloat16).clone())


# ------------------------------------------------------ dispatcher (3A)
def test_bound_dense_without_override_enters_eager_not_native():
    """_install_dense-style binding: an active tagged call without an override
    goes to dense_eager_attention_forward, never the installed native SDPA."""
    import dllm.attention.blasst.integration as blasst
    from dllm.attention.blasst.core import Blasst2DConfig

    entered = []
    registry = {'sdpa': lambda module, *a, **k: entered.append('native') or ('native', None)}
    module = torch.nn.Linear(1, 1)
    saved = blasst.dense_eager_attention_forward
    blasst.dense_eager_attention_forward = lambda module, *a, **k: entered.append('eager') or ('eager', None)
    try:
        registry['sdpa'](module)
        assert entered == ['native']                       # unbound: native
        binding = blasst._attach_registry(
            module, [module], registry, Blasst2DConfig(enable_blasst_2d=True, apply_blasst_mask=False),
            None, mask_token_id=None, pad_token_id=None, query_ids_extractor=None,
            filter_special_query_ids=False, call_selector=None, dense_kv_prefix_extractor=None,
            sweep_lambdas=(), attention_observer=None)
        binding.runtime.active_call = True
        entered.clear()
        registry['sdpa'](module)
        assert entered == ['eager']                        # the mislabeled "native_dense"
        binding.runtime.force_native_attention = True
        entered.clear()
        registry['sdpa'](module)
        assert entered == ['native']                       # qualified force-original
        binding.close()
        entered.clear()
        registry['sdpa'](module)
        assert entered == ['native']                       # untouched after close
    finally:
        blasst.dense_eager_attention_forward = saved


# ---------------------------------------------- real ScoreCache phases (3E)
def identity(layer=0):
    return Identity(0, 0, 0, layer, 1, 4, 2, 64, 256, 128, 192, 0, 256, 1., 'bf16', 'cpu', (), 7)


def test_v8_m3_held_search_from_anchor_zero_can_never_find_a_held_step():
    cache = ScoreCache(8, 2)
    cache.publish_scores(identity(), 0, object())
    cache.publish_decision(identity(), 0, object())
    held = [step for step in range(3, 11) if not cache.plan(identity(), step).decision_refresh]
    assert held == []                       # the v8 helper's search space is all refreshes


def test_m3_anchor0_then_step1_is_a_genuinely_held_phase():
    cache = ScoreCache(8, 2)
    plan = cache.plan(identity(), 0)
    assert plan.score_refresh and plan.decision_refresh
    cache.publish_scores(identity(), 0, object())
    cache.publish_decision(identity(), 0, object())
    held = cache.plan(identity(), 1)
    assert not held.score_refresh and not held.decision_refresh and held.reason == 'held_decision'
    # and the next due decision really is a decision refresh
    assert cache.plan(identity(), 2).reason == 'decision_schedule'


def test_m1_anchor0_then_step1_is_an_ordinary_decision_on_age1_scores():
    cache = ScoreCache(8, 1)
    cache.publish_scores(identity(), 0, object())
    cache.publish_decision(identity(), 0, object())
    plan = cache.plan(identity(), 1)
    assert (plan.score_refresh, plan.decision_refresh, plan.score_age) == (False, True, 1)


# ------------------------------------------------------- telemetry (3E)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_reset_router_clears_telemetry_so_repetitions_do_not_time_list_growth():
    from tests.test_numerical_reuse_summary_memory import call, inputs, make_router
    router, modules, _ = make_router()
    generator = torch.Generator(device='cuda').manual_seed(2)
    x = inputs(generator, 0)
    try:
        lengths = []
        for _ in range(3):
            reset_router(router)
            router.begin_step(0, 0)
            before = telemetry(router)
            call(router, modules[0], x)
            router.begin_step(0, 1)
            call(router, modules[0], x)
            after = telemetry(router)
            lengths.append((before['pending'], after['pending'], after['call_metadata'],
                            after['calls'] - before['calls']))
        assert lengths == [(0, 2, 2, 2)] * 3
    finally:
        router.close()
