"""Untimed physical-work counter twin for v20 native-legal attention.

Install only around a separate diagnostic replay. The wrapper reads completed
device bitmaps on the host after each real attention call, so it must never be
used in accepted timing or scored generation. Counts are *legal q-k pairs*,
not allocated Q128/KV64 tiles, bytes, active GPU cycles or Q/K/V projections.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

Q_TILE = 128
K_TILE = 64
SEGMENTS = ('static_prefix', 'current_canvas', 'whole')


def pair_weights(nq: int, nk: int, prefix: int) -> list[list[dict[str, int]]]:
    """Legal pair weights per physical tile for native bidirectional mask=None.

    A KV64 tile straddling the frozen-prefix boundary is split by token count;
    short Q/K tails contribute only their real rows/columns.
    """
    if any(type(x) is not int for x in (nq, nk, prefix)) or not (0 < nq <= nk and prefix == nk - nq):
        raise ValueError('invalid native Q128/KV64 prefix and canvas geometry')
    result = []
    for qb in range((nq + Q_TILE - 1) // Q_TILE):
        rows = min(Q_TILE, nq - qb * Q_TILE)
        tiles = []
        for kt in range((nk + K_TILE - 1) // K_TILE):
            k0, k1 = kt * K_TILE, min(nk, (kt + 1) * K_TILE)
            static = rows * max(0, min(k1, prefix) - k0)
            current = rows * max(0, k1 - max(k0, prefix))
            tiles.append(dict(static_prefix=static, current_canvas=current, whole=static + current))
        result.append(tiles)
    return result


def count_bitmap(skipped, eligible, *, nq: int, nk: int, prefix: int,
                 phase: str, head_dim: int, full_current_qk: bool | None = None) -> dict:
    """Count exact legal-pair weighted QK/PV work on an observed physical bitmap.

    `skipped` and `eligible` are nested bool [B,H,QB,KT] data from an untimed
    device-to-host transfer. A historical A observes full current QK before
    routing; D/H read the historical score/bitmap and issue current QK only on
    retained support. A control bootstrap can supply full_current_qk=False:
    its all-kept consumer still computes every legal pair without materializing
    a separate full-score tensor.
    """
    if phase not in ('A', 'D', 'H') or type(head_dim) is not int or head_dim <= 0:
        raise ValueError('explicit A/D/H phase and positive head dimension required')
    if full_current_qk is None:
        full_current_qk = phase == 'A'
    if type(full_current_qk) is not bool:
        raise ValueError('full_current_qk must be a boolean observation')
    weights = pair_weights(nq, nk, prefix)
    qb, kt = len(weights), len(weights[0])
    if not isinstance(skipped, list) or not skipped or len(skipped) != len(eligible):
        raise ValueError('bitmap batch dimensions differ')
    counters = {segment: dict(eligible_pairs=0, skipped_qk_pairs=0,
                              executed_qk_pairs=0, skipped_pv_pairs=0,
                              executed_pv_pairs=0) for segment in SEGMENTS}
    heads = None
    for batch_skip, batch_eligible in zip(skipped, eligible):
        if not isinstance(batch_skip, list) or not batch_skip or len(batch_skip) != len(batch_eligible):
            raise ValueError('bitmap head dimensions differ')
        if heads is None:
            heads = len(batch_skip)
        elif heads != len(batch_skip):
            raise ValueError('bitmap head count differs across batch')
        for head_skip, head_eligible in zip(batch_skip, batch_eligible):
            if len(head_skip) != qb or len(head_eligible) != qb:
                raise ValueError('bitmap Q128 dimension differs')
            for qtile in range(qb):
                if len(head_skip[qtile]) != kt or len(head_eligible[qtile]) != kt:
                    raise ValueError('bitmap KV64 dimension differs')
                for ktile in range(kt):
                    drop, legal = head_skip[qtile][ktile], head_eligible[qtile][ktile]
                    if type(drop) is not bool or type(legal) is not bool or not legal or drop and not legal:
                        raise ValueError('native-legal bitmap must cover every physical legal tile')
                    for segment in SEGMENTS:
                        pairs = weights[qtile][ktile][segment]
                        row = counters[segment]
                        row['eligible_pairs'] += pairs
                        row['skipped_pv_pairs'] += pairs if drop else 0
                        row['executed_pv_pairs'] += 0 if drop else pairs
                        qk_drop = drop and not full_current_qk
                        row['skipped_qk_pairs'] += pairs if qk_drop else 0
                        row['executed_qk_pairs'] += 0 if qk_drop else pairs
    for row in counters.values():
        row['executed_qk_multiply_accumulates'] = row['executed_qk_pairs'] * head_dim
        row['executed_pv_multiply_accumulates'] = row['executed_pv_pairs'] * head_dim
    partial_tiles = [dict(q_tile=q_index, k_tile=k_index, **tile)
                     for q_index, row in enumerate(weights)
                     for k_index, tile in enumerate(row)
                     if tile['whole'] != Q_TILE * K_TILE or
                     (tile['static_prefix'] and tile['current_canvas'])]
    return dict(phase=phase, full_current_qk=full_current_qk, batch=len(skipped), heads=heads,
                bitmap_shape=[len(skipped), heads, qb, kt],
                query_tokens=nq, key_tokens=nk, prefix_tokens=prefix,
                head_dim=head_dim, by_segment=counters, partial_tile_pair_weights=partial_tiles,
                projection_skipping_measured=False,
                unit='native-legal q-k pairs; multiply-accumulates multiply by head_dim')


def _host_bitmap(tensor):
    # This synchronization/transfer is permitted only in the explicit twin.
    return tensor.detach().cpu().tolist()


class CounterTwin:
    """Callable runtime override delegating the real attention call unchanged."""

    def __init__(self, delegate, router, *, retain_support=False, qkv_digest=None,
                 operator_probe=None):
        self.delegate, self.router = delegate, router
        self.rows = []
        self.retain_support, self.qkv_digest = retain_support, qkv_digest
        self.support_calls = []
        self.control_observation_steps = {}
        self.operator_probe = operator_probe

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _fresh_call(self, module, q, k, v, mask, **kwargs):
        """Read one native-legal fresh kernel bitmap only in this untimed twin."""
        router = self.router
        if router.support_geometry != 'native_legal':
            raise ValueError('fresh counter requires native-legal support')
        pending = router.pending
        before, old_collect, old_calls = len(pending), router.collect, router.calls
        router.collect = True
        try:
            result = self.delegate(module, q, k, v, mask, **kwargs)
            if len(pending) != before + 1 or router.calls != old_calls + 1:
                raise ValueError('fresh counter expected exactly one new bitmap and call')
            layer, step, kind, prefix, skipped, eligible = pending[before]
            nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
            if (layer != int(module.layer_idx) or prefix != nk - nq or v.shape != k.shape or
                    kind != ('local' if bool(module.is_sliding) else 'global')):
                raise ValueError('fresh bitmap/source geometry drift')
            row = count_bitmap(_host_bitmap(skipped), _host_bitmap(eligible),
                               nq=nq, nk=nk, prefix=prefix, phase='A',
                               head_dim=d, full_current_qk=True)
            if row['bitmap_shape'][:2] != [q.shape[0], q.shape[1]]:
                raise ValueError('fresh bitmap batch/head dimensions differ')
            row.update(layer=layer, kind=kind, canvas=None, decoder_call=step,
                       method='fresh_T', anchor_step=step, decision_step=step,
                       numeric_age=0, decision_age=0)
            self.rows.append(row)
            return result
        finally:
            # Do not change the router's persistent collection contract.
            router.collect = old_collect
            del pending[before:]

    def __call__(self, module, q, k, v, mask, **kwargs):
        # Attention infers causal=True for a multi-token canvas when this kwarg
        # is absent. The qualified native decoder passes False explicitly.
        if mask is not None or kwargs.get('is_causal') is not False:
            raise ValueError('counter twin requires explicit bidirectional is_causal=False and mask=None')
        if getattr(self.router, 'support_geometry', None) == 'native_legal':
            return self._fresh_call(module, q, k, v, mask, **kwargs)
        owner = getattr(self.router, 'owner', self.router)
        if owner.support != 'native_mask':
            raise ValueError('counter twin requires native-legal support')
        if owner.output_mode != 'historical_route_preqk_current_output':
            raise ValueError('counter twin requires the pre-QK current-output consumer')
        layer = int(module.layer_idx)
        old_score = getattr(self.router, 'score_calls', None)
        old_decision = getattr(self.router, 'decision_calls', None)
        old_held = getattr(self.router, 'held_calls', None)
        old_bootstrap = getattr(self.router, 'bootstrap_calls', None)
        old_observation = getattr(self.router, 'observation_calls', None)
        result = self.delegate(module, q, k, v, mask, **kwargs)
        if old_score is not None:
            score = self.router.score_calls - old_score
            decision = self.router.decision_calls - old_decision
            held = self.router.held_calls - old_held
            if (score, decision, held) == (1, 1, 0):
                phase, full_qk = 'A', True
            elif (score, decision, held) == (0, 1, 0):
                phase, full_qk = 'D', False
            elif (score, decision, held) == (0, 0, 1):
                phase, full_qk = 'H', False
            else:
                raise ValueError('router phase counters do not identify one A/D/H call')
            entry = self.router.cache.entries[layer]
            bitmap = entry.decision
            skipped, eligible = bitmap.skipped, bitmap.eligible
            anchor_step, decision_step = entry.score_step, entry.decision_step
            if (type(anchor_step) is not int or type(decision_step) is not int or
                    not 0 <= anchor_step <= owner.step or not 0 <= decision_step <= owner.step):
                raise ValueError('numerical score/decision anchor age is unavailable or invalid')
            numeric_age, decision_age = owner.step-anchor_step, owner.step-decision_step
        elif old_bootstrap is not None:
            bootstrap = self.router.bootstrap_calls - old_bootstrap
            observation = self.router.observation_calls - old_observation
            held = self.router.held_calls - old_held
            if (bootstrap, observation, held) == (1, 0, 0):
                phase, full_qk = 'A', False
                anchor_step = decision_step = numeric_age = decision_age = None
                b, h, nq = q.shape[:3]
                nk = k.shape[-2]
                shape_q, shape_k = (nq + Q_TILE - 1) // Q_TILE, (nk + K_TILE - 1) // K_TILE
                skipped = [[[[False] * shape_k for _ in range(shape_q)] for _ in range(h)] for _ in range(b)]
                eligible = [[[[True] * shape_k for _ in range(shape_q)] for _ in range(h)] for _ in range(b)]
            elif (bootstrap, observation, held) == (0, 1, 0):
                phase, full_qk = 'A', True
                _, skipped, eligible = self.router.maps[layer]
                key = (owner.canvas, owner.epoch, layer)
                self.control_observation_steps[key] = owner.step
                anchor_step = decision_step = owner.step
                numeric_age = decision_age = 0
            elif (bootstrap, observation, held) == (0, 0, 1):
                phase, full_qk = 'H', False
                _, skipped, eligible = self.router.maps[layer]
                key = (owner.canvas, owner.epoch, layer)
                observed_step = self.control_observation_steps.get(key)
                if type(observed_step) is not int or not 0 <= observed_step <= owner.step:
                    raise ValueError('held control lacks a same-canvas observed bitmap age')
                anchor_step = decision_step = observed_step
                numeric_age = decision_age = owner.step-observed_step
            else:
                raise ValueError('control phase counters do not identify one A/H call')
        else:
            raise ValueError('unsupported counter twin router')
        prefix = owner.sources[layer][3]
        nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
        if nk != prefix + nq or v.shape != k.shape:
            raise ValueError('counter twin source/canvas geometry differs')
        physical_skipped, physical_eligible = skipped, eligible
        if not isinstance(skipped, list):
            skipped = _host_bitmap(skipped)
            eligible = _host_bitmap(eligible)
        row = count_bitmap(skipped, eligible, nq=nq, nk=nk, prefix=prefix,
                           phase=phase, head_dim=d, full_current_qk=full_qk)
        if row['bitmap_shape'][:2] != [q.shape[0], q.shape[1]]:
            raise ValueError('physical bitmap batch/head dimensions differ from current query')
        row.update(layer=layer, kind='local' if bool(module.is_sliding) else 'global',
                   canvas=owner.canvas, decoder_call=owner.step,
                   method='historical_or_same_consumer_control',
                   anchor_step=anchor_step, decision_step=decision_step,
                   numeric_age=numeric_age, decision_age=decision_age)
        self.rows.append(row)
        if self.operator_probe is not None and old_score is not None:
            self.operator_probe.observe(self.router, module, q, k, v,
                                        bitmap.skipped, bitmap.eligible,
                                        phase=phase, canvas=owner.canvas, step=owner.step,
                                        scale=kwargs.get('scaling'), actual_output=result)
        elif self.operator_probe is not None and old_bootstrap is not None and hasattr(physical_skipped, 'detach'):
            # Held-map reference: observe/H use the same prepared consumer on
            # the actual current QKV and recorded physical support. Bootstrap
            # has only synthetic all-kept lists and is intentionally omitted.
            self.operator_probe.observe(owner, module, q, k, v, physical_skipped, physical_eligible,
                                        phase=phase, canvas=owner.canvas, step=owner.step,
                                        scale=kwargs.get('scaling'), actual_output=result)
        if self.retain_support:
            device_skipped = bitmap.skipped if old_score is not None else self.router.maps[layer][1]
            device_eligible = bitmap.eligible if old_score is not None else self.router.maps[layer][2]
            if not hasattr(device_skipped, 'detach') or not hasattr(device_eligible, 'detach'):
                raise ValueError('prepared floor requires actual device support tensors')
            self.support_calls.append(dict(layer=layer, canvas=owner.canvas, decoder_call=owner.step,
                                           q_shape=tuple(q.shape), k_shape=tuple(k.shape),
                                           prefix=prefix, skipped=device_skipped.detach().clone(),
                                           eligible=device_eligible.detach().clone(),
                                           qkv_digest=(self.qkv_digest(q, k, v)
                                                       if self.qkv_digest is not None else None)))
        return result

    def summary(self):
        by_phase_kind = defaultdict(lambda: {segment: dict(eligible_pairs=0,
            skipped_qk_pairs=0, executed_qk_pairs=0, skipped_pv_pairs=0,
            executed_pv_pairs=0, executed_qk_multiply_accumulates=0,
            executed_pv_multiply_accumulates=0) for segment in SEGMENTS})
        for row in self.rows:
            aggregate = by_phase_kind[(row['phase'], row['kind'])]
            for segment, counts in row['by_segment'].items():
                for key, value in counts.items():
                    aggregate[segment][key] += value
        return dict(schema='v20_untimed_counter_twin_v1', calls=len(self.rows),
                    by_phase_kind={f'{phase}/{kind}': counts for (phase, kind), counts in
                                   sorted(by_phase_kind.items())},
                    projection_skipping_measured=False,
                    accepted_timing=False, rows=self.rows)


@contextmanager
def install_counter_twin(binding, router, *, explicit_untimed: bool = False,
                         retain_support: bool = False, qkv_digest=None, operator_probe=None):
    """Profiler API: ``with install_counter_twin(binding, router, explicit_untimed=True) as twin``.

    The binding must already be installed; this wraps its exact prior override
    (including GLOBAL mask guards), and restores it on exit. No production
    installer invokes this automatically.
    """
    if explicit_untimed is not True:
        raise ValueError('counter twin is only permitted in an explicit untimed pass')
    runtime = binding.runtime
    delegate = runtime.attention_override
    twin = CounterTwin(delegate, router, retain_support=retain_support,
                       qkv_digest=qkv_digest, operator_probe=operator_probe)
    runtime.attention_override = twin
    try:
        yield twin
    finally:
        runtime.attention_override = delegate


class PreparedSupportFloor:
    """Same numerical consumer with each call's observed support supplied for free.

    This is a separate diagnostic pass, not a deployable selector or accepted
    production timing. Every replay must consume the full frozen call sequence.
    """

    def __init__(self, router, support_calls, *, qkv_digest=None):
        self.router, self.support_calls, self.qkv_digest = router, support_calls, qkv_digest
        self.cursor = 0
        self.qkv_matches = []

    def reset(self):
        self.cursor = 0
        self.qkv_matches = []

    def assert_complete(self):
        if self.cursor != len(self.support_calls):
            raise ValueError('prepared support replay missed frozen attention calls')

    def __call__(self, module, q, k, v, mask, *, dropout=0., scaling=None,
                 is_causal=None, **kwargs):
        if mask is not None or is_causal is not False or dropout or module.training:
            raise ValueError('prepared support floor requires native bidirectional inference')
        if self.cursor >= len(self.support_calls):
            raise ValueError('prepared support replay exceeded frozen call sequence')
        item = self.support_calls[self.cursor]
        self.cursor += 1
        owner = getattr(self.router, 'owner', self.router)
        layer = int(module.layer_idx)
        actual = (layer, owner.canvas, owner.step, tuple(q.shape), tuple(k.shape), owner.sources[layer][3])
        expected = (item['layer'], item['canvas'], item['decoder_call'],
                    item['q_shape'], item['k_shape'], item['prefix'])
        if actual != expected or v.shape != k.shape:
            raise ValueError('prepared support call/source geometry drift')
        if self.qkv_digest is not None:
            self.qkv_matches.append(self.qkv_digest(q, k, v) == item['qkv_digest'])
        from .cached_executor import fused_guard
        scale = float(scaling) if scaling is not None else q.shape[-1] ** -.5
        owner._mask_present = False
        result = owner._consume(q, k, v, item['skipped'], item['eligible'], scale, None, False)
        returned = result.output.transpose(1, 2).contiguous()
        fused_guard(returned, result.invalid_scores)
        return returned, None


@contextmanager
def install_prepared_support_floor(binding, router, support_calls, *, explicit_untimed=False,
                                   qkv_digest=None):
    if explicit_untimed is not True or not support_calls:
        raise ValueError('prepared support floor requires an explicit untimed support capture')
    runtime = binding.runtime
    delegate = runtime.attention_override
    floor = PreparedSupportFloor(router, support_calls, qkv_digest=qkv_digest)
    runtime.attention_override = floor
    try:
        yield floor
    finally:
        runtime.attention_override = delegate
