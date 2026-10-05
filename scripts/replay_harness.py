"""v9 replay machinery: per-repetition state restoration and path proofs.

What v8's harness got wrong (v9 spec section 3), and what replaces it here:

* RNG was restored once around a whole timed series, so every warmup and
  repetition consumed different random draws. ``StepSnapshot.restore`` puts
  CPU/CUDA RNG back before EACH repetition, outside the timed region.
* ``clone_kwargs`` cloned tensors only; the native sampler, the stateful
  diffusion stopping criterion and the T controller were shared and kept
  mutating. ``ObjectState`` snapshots every attribute the LOADED objects
  actually carry (discovered with ``vars()``, not assumed), recursing into
  plain-Python children, and records what it could not snapshot.
* The capture ran under ``observe(model, state)``, which wraps
  ``kwargs['sampler']`` in ``query_adaptive.Sampler``; that wrapper (and its
  capture-time State) leaked into every replay. ``unwrap_sampler`` strips it
  and ``capture_window`` installs its own wrapper OUTSIDE observe, so the
  sampler it records is the native one.
* Input digests were computed twice from the same saved kwargs before the
  series. ``StepSnapshot.digest`` is recomputed after each restore and each
  replay's OUTPUT is digested, so drift is detected where it would happen.
* Replay fixtures were copied inside the timed interval. ``prepare`` builds
  them before the timer starts.

Digests are bit-level: floating tensors are reinterpreted as integers before
the order-sensitive reductions, so -0/+0 or NaN-payload changes are visible.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import time
from types import MethodType
from typing import Any, Callable

import torch

_INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16,
             torch.float32: torch.int32, torch.float64: torch.int64}
PRIMITIVE = (int, float, bool, str, bytes, type(None))


def tensor_digest(value: torch.Tensor) -> tuple:
    """Bit-sensitive, order-sensitive digest (shape, dtype, two reductions)."""
    flat = value.detach().contiguous().reshape(-1)
    if flat.dtype in _INT_VIEW:
        flat = flat.view(_INT_VIEW[flat.dtype])
    elif flat.dtype == torch.bool:
        flat = flat.to(torch.int8)
    wide = flat.to(torch.float64)
    index = torch.arange(1, wide.numel() + 1, device=wide.device, dtype=torch.float64)
    return (tuple(value.shape), str(value.dtype), float(wide.sum()),
            float((wide * torch.remainder(index, 65521.)).sum()))


def value_digest(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_digest(value)
    if isinstance(value, torch.Generator):
        return ('generator', hashlib.sha256(value.get_state().numpy().tobytes()).hexdigest())
    if isinstance(value, PRIMITIVE):
        return value
    if isinstance(value, (list, tuple)):
        return [value_digest(item) for item in value]
    if isinstance(value, dict):
        return {str(key): value_digest(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if hasattr(value, '__dict__'):
        return {'type': type(value).__qualname__,
                **{key: value_digest(item) for key, item in sorted(vars(value).items())}}
    return ('opaque', type(value).__qualname__)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value_digest(value), sort_keys=True, default=str)
                          .encode()).hexdigest()


class ObjectState:
    """Snapshot/restore of an object's OWN attributes, discovered at runtime.

    Tensors are cloned at snapshot and re-cloned at restore (a replay can then
    mutate them in place without touching the saved copy). ``torch.Generator``
    state is saved via get_state/set_state. Plain-Python children are recursed
    into up to ``depth``. Anything else (modules, functions, configs) is kept by
    reference and listed in ``shared`` so the report says what was NOT copied.
    Attributes created after the snapshot are deleted on restore.
    """

    def __init__(self, obj: Any, *, depth: int = 2, exclude: tuple[str, ...] = ()):
        self.obj, self.exclude = obj, set(exclude)
        self.shared: list[str] = []
        self.saved = {key: self._snap(key, value, depth) for key, value in vars(obj).items()
                      if key not in self.exclude}
        self.layout = {key: type(value).__qualname__ for key, value in vars(obj).items()}

    def _snap(self, name: str, value: Any, depth: int):
        if torch.is_tensor(value):
            return ('tensor', value.detach().clone())
        if isinstance(value, torch.Generator):
            return ('generator', value, value.get_state())
        if isinstance(value, PRIMITIVE):
            return ('value', value)
        if isinstance(value, (list, tuple)) and depth > 0:
            return ('seq', type(value), [self._snap(f'{name}[{i}]', item, depth - 1)
                                         for i, item in enumerate(value)])
        if isinstance(value, dict) and depth > 0:
            return ('dict', {key: self._snap(f'{name}.{key}', item, depth - 1)
                             for key, item in value.items()})
        if hasattr(value, '__dict__') and depth > 0 and not isinstance(value, torch.nn.Module):
            return ('object', ObjectState(value, depth=depth - 1))
        self.shared.append(f'{name}:{type(value).__qualname__}')
        return ('shared', value)

    @staticmethod
    def _thaw(saved):
        kind = saved[0]
        if kind == 'tensor':
            return saved[1].clone()
        if kind == 'generator':
            saved[1].set_state(saved[2])
            return saved[1]
        if kind in ('value', 'shared'):
            return saved[1]
        if kind == 'seq':
            items = [ObjectState._thaw(item) for item in saved[2]]
            return saved[1](items) if saved[1] in (list, tuple) else _rebuild(saved[1], items)
        if kind == 'dict':
            return {key: ObjectState._thaw(item) for key, item in saved[1].items()}
        if kind == 'object':
            return saved[1].restore()
        raise AssertionError(kind)

    def restore(self, target: Any = None):
        target = self.obj if target is None else target
        for key in [key for key in vars(target) if key not in self.saved and key not in self.exclude]:
            delattr(target, key)
        for key, saved in self.saved.items():
            setattr(target, key, self._thaw(saved))
        return target

    def all_shared(self) -> list[str]:
        out = list(self.shared)
        for saved in self.saved.values():
            if saved[0] == 'object':
                out.extend(saved[1].all_shared())
        return out


def _rebuild(kind, items):
    try:
        return kind(items)
    except TypeError:
        rebuilt = kind()
        rebuilt.extend(items)
        return rebuilt


def unwrap_sampler(sampler: Any) -> Any:
    """Strip ``query_adaptive.Sampler`` wrappers (and their State) to the native one."""
    from experiments.value_direction_hopper.query_adaptive import Sampler
    while isinstance(sampler, Sampler):
        sampler = sampler.inner
    return sampler


STATEFUL_KWARGS = ('sampler', 'diffusion_stopping_criteria', 'logits_processor')


def execution_context(model: Any) -> dict[str, Any]:
    """Flags that must match between production and replay, read at the boundary."""
    step = getattr(type(model), '_denoising_step')
    generate = getattr(type(model), 'generate')
    return dict(inference_mode=torch.is_inference_mode_enabled(),
                grad_enabled=torch.is_grad_enabled(),
                autocast_cuda=torch.is_autocast_enabled('cuda') if torch.cuda.is_available() else None,
                autocast_cpu=torch.is_autocast_enabled('cpu'),
                current_stream=(torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else None),
                default_stream=(torch.cuda.default_stream().cuda_stream if torch.cuda.is_available() else None),
                attn_implementation=getattr(getattr(model, 'config', None), '_attn_implementation', None),
                text_attn_implementation=getattr(getattr(getattr(model, 'config', None), 'text_config', None),
                                                 '_attn_implementation', None),
                denoising_step_is_wrapped=hasattr(step, '__wrapped__'),
                denoising_step_instance_override='_denoising_step' in vars(model),
                generate_is_wrapped=hasattr(generate, '__wrapped__'),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


class StepSnapshot:
    """Everything one native denoising step reads that a replay could perturb."""

    def __init__(self, kwargs: dict[str, Any], *, controller: Any = None):
        self.tensors = {key: value.detach().clone() for key, value in kwargs.items() if torch.is_tensor(value)}
        self.objects: dict[str, list[ObjectState]] = {}
        self.plain: dict[str, Any] = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                continue
            if key == 'sampler':
                value = unwrap_sampler(value)
            if key in STATEFUL_KWARGS and value is not None:
                # A LogitsProcessorList is a list: snapshot each processor.
                self.objects[key] = ([ObjectState(item) for item in value]
                                     if isinstance(value, (list, tuple)) else [ObjectState(value)])
            self.plain[key] = value
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        # T controller (query_adaptive.State): its router reference is live
        # production state, never part of the captured prior-step snapshot.
        self.controller = (None if controller is None else
                           ObjectState(controller, depth=2, exclude=('router',)))

    def prepare(self, controller: Any = None) -> dict[str, Any]:
        """Restore state and build fresh kwargs. Call OUTSIDE any timed region."""
        for states in self.objects.values():
            for state in states:
                state.restore()
        if self.controller is not None and controller is not None:
            self.controller.restore(controller)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        kwargs = dict(self.plain)
        kwargs.update({key: value.clone() for key, value in self.tensors.items()})
        return kwargs

    @staticmethod
    def digest(kwargs: dict[str, Any], controller: Any = None) -> str:
        """Input + mutable-state digest, recomputed per repetition."""
        payload = {key: value_digest(value) for key, value in kwargs.items()
                   if key not in ('decoder_forward', 'past_key_values')}
        payload['past_key_values'] = cache_digest(kwargs.get('past_key_values'))
        payload['rng_cpu'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        if torch.cuda.is_available():
            payload['rng_cuda'] = [hashlib.sha256(s.numpy().tobytes()).hexdigest()
                                   for s in torch.cuda.get_rng_state_all()]
        if controller is not None:
            payload['controller'] = {key: value_digest(value) for key, value in sorted(vars(controller).items())
                                     if key != 'router'}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


    def inventory(self) -> dict[str, Any]:
        """What was snapshotted per stateful object, and what stayed shared."""
        return {key: [dict(type=type(state.obj).__qualname__, attributes=state.layout,
                           shared_not_copied=state.all_shared()) for state in states]
                for key, states in self.objects.items()}


def cache_digest(cache: Any) -> Any:
    if cache is None:
        return None
    layers = getattr(cache, 'layers', None)
    if layers is None:
        return ('opaque', type(cache).__qualname__)
    return [(tensor_digest(layer.keys), tensor_digest(layer.values)) if getattr(layer, 'keys', None) is not None
            else None for layer in layers]


def output_digest(result: Any) -> str:
    return fingerprint(list(result) if isinstance(result, (tuple, list)) else result)


@contextmanager
def dispatch_spy(model: Any):
    """Count which attention implementation each decoder attention call enters.

    Wraps, for the duration only, the registry's CURRENT ``sdpa`` entry (the
    untouched native function when nothing is bound, the BLASST dispatcher when
    a binding is installed), the ``dense_eager_attention_forward`` and
    ``blasst_2d_attention_forward`` symbols that dispatcher resolves at call
    time, and ``Attention.__call__``. Calls that entered the registry but none
    of the named non-native branches went to the dispatcher's captured native
    function. Bounded probe: never used inside accepted timing.
    """
    import importlib
    import dllm.attention.blasst.integration as blasst
    from experiments.numerical_qk_reuse import integration as numerical
    from transformers.integrations.sdpa_attention import sdpa_attention_forward as native
    modeling = importlib.import_module(type(model).__module__.replace('generation_', 'modeling_'))
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    entered = registry['sdpa']
    counts = dict(registry_entry=f'{entered.__module__}.{entered.__qualname__}',
                  registry_entry_is_native=entered is native,
                  registry_sdpa=0, dense_eager=0, blasst_masked=0, numerical_selector=0)
    saved = dict(eager=blasst.dense_eager_attention_forward, masked=blasst.blasst_2d_attention_forward,
                 call=numerical.Attention.__call__)

    def registry_spy(*args, **kwargs):
        counts['registry_sdpa'] += 1
        return entered(*args, **kwargs)

    def eager_spy(*args, **kwargs):
        counts['dense_eager'] += 1
        return saved['eager'](*args, **kwargs)

    def masked_spy(*args, **kwargs):
        counts['blasst_masked'] += 1
        return saved['masked'](*args, **kwargs)

    def call_spy(self, *args, **kwargs):
        counts['numerical_selector'] += 1
        return saved['call'](self, *args, **kwargs)

    registry['sdpa'] = registry_spy
    blasst.dense_eager_attention_forward = eager_spy
    blasst.blasst_2d_attention_forward = masked_spy
    numerical.Attention.__call__ = call_spy
    try:
        yield counts
    finally:
        registry['sdpa'] = entered
        blasst.dense_eager_attention_forward = saved['eager']
        blasst.blasst_2d_attention_forward = saved['masked']
        numerical.Attention.__call__ = saved['call']
        counts['native_sdpa'] = (counts['registry_sdpa'] if counts['registry_entry_is_native'] else
                                 counts['registry_sdpa'] - counts['dense_eager']
                                 - counts['blasst_masked'] - counts['numerical_selector'])


def assert_native_path(counts: dict) -> None:
    """Fail if a 'native' measurement touched eager attention or our selector."""
    if not counts['registry_entry_is_native'] or counts['dense_eager'] or counts['blasst_masked'] \
            or counts['numerical_selector'] or counts['registry_sdpa'] == 0:
        raise AssertionError(f'native baseline did not run the untouched native SDPA path: {counts}')


def reset_router(router: Any) -> None:
    """Return a numerical router to 'no history' without touching semantics.

    Clears score/decision history, summaries, V-sketch leases AND the optional
    per-call telemetry (``pending``/``call_metadata``) that ``reset()`` of the
    old harness left growing.
    """
    router.cache.clear()
    router.valid_keys.clear()
    router.summaries.clear()
    router.sketches.entries.clear()
    router.pending.clear()
    router.call_metadata.clear()
    router.canvas, router.step = -1, -1


def telemetry(router: Any) -> dict[str, int]:
    return dict(pending=len(router.pending), call_metadata=len(router.call_metadata),
                calls=router.calls, score_calls=router.score_calls,
                decision_calls=router.decision_calls, held_calls=router.held_calls,
                preqk_calls=router.preqk_calls, summary_hits=router.summary_hits,
                summary_builds=router.summary_builds, summary_misses=router.summary_misses)


def timed_rows(run: Callable[[dict], Any], prepare: Callable[[], dict], *, warmup: int, reps: int,
               check: Callable[[dict, Any], dict] | None = None) -> dict[str, Any]:
    """prepare() (untimed) -> synchronize -> timed run(kwargs) -> synchronize.

    Both CUDA-event span and host wall are recorded: the span is NOT GPU-active
    time (it contains any host launch gaps). ``check`` runs untimed after each
    repetition and returns per-repetition evidence (digests, telemetry).
    """
    first, samples, walls, evidence = None, [], [], []
    for index in range(warmup + reps):
        fixture = prepare()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        host = time.perf_counter()
        start.record()
        result = run(fixture)
        end.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - host) * 1e3
        span = start.elapsed_time(end)
        row = check(fixture, result) if check is not None else {}
        row.update(index=index, warmup=index < warmup, event_ms=span, wall_ms=wall)
        evidence.append(row)
        if index == 0:
            first = span
        if index >= warmup:
            samples.append(span)
            walls.append(wall)
    ordered = sorted(samples)
    import statistics
    return dict(median_ms=statistics.median(ordered), min_ms=ordered[0], max_ms=ordered[-1],
                wall_median_ms=statistics.median(walls), first_observation_ms=first,
                reps=len(samples), warmup=warmup, per_repetition=evidence,
                timing_note='event span includes host launch gaps; not GPU-active union')
