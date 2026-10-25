"""Unoptimized selector-only layer29 execution using current paged Q/K/V."""
import hashlib

import torch

from selector import choose_unit, mass_lower_bound


class Runtime:
    def __init__(self, mode, audit):
        self.mode, self.audit = mode, audit
        self.active = False
        self.begin()
        self.active = False

    def begin(self):
        self.active = True
        self.phase = None
        self.prefix = None
        self.step = 0
        self.keep = None
        self.previous_q = None
        self.last_winner = None
        self.stable = None
        self.margin = None
        self.certificates = []
        self.records, self.events, self.banks = [], [], []
        self.refreshes = self.held = self.fallback_units = self.certificate_failures = 0
        self.skipped = self.eligible = 0

    def native_sample(self, result, args, committing):
        if not self.active or committing or self.mode != 'margin_guard_weighted_bias':
            return
        winner = args[6][args[1]].reshape(-1).detach().clone()
        self.stable = (winner == self.last_winner) if self.last_winner is not None else torch.zeros_like(winner, dtype=torch.bool)
        self.last_winner = winner
        top = result.topk(2, dim=-1).values.float().reshape(-1, 2)
        self.margin = (top[:, 0] - top[:, 1]).detach()

    def receipt(self):
        if self.events:
            torch.cuda.synchronize()
        times = {}
        for phase, start, end in self.events:
            row = times.setdefault(phase, dict(calls=0, gpu_ms=0.))
            row['calls'] += 1
            row['gpu_ms'] += start.elapsed_time(end)
        return dict(selector=self.mode, scope='GLOBAL layer29; all16 heads; queryblocks0/1; Q128xKV64',
            selector_refresh_calls=self.refreshes, held_mask_calls=self.held,
            carried_mask_calls=0, fallback_units=self.fallback_units,
            certificate_failures=self.certificate_failures,
            target_eligible_tiles=self.eligible, target_skipped_tiles=self.skipped,
            target_kept_tiles=self.eligible - self.skipped,
            selector_extra_model_forwards=0, current_qkv=True,
            cached_old_kv=False, cached_attention_outputs=False,
            selector_phase_gpu=times, physical_calls=self.records if self.audit else None,
            first_two_candidate_banks=self.banks if self.audit else None)

    def install(self):
        from vllm.model_executor.models import diffusion_gemma as dg
        from vllm.v1.attention.backends import flash_attn as fa
        original_prepare = dg.DiffusionGemmaModelState.prepare_attn
        original_forward = fa.FlashAttentionImpl.forward
        rt = self

        def prepare(state, batch, mode, tables, slots, groups, config, for_capture=False, ubatch_idx=0):
            rt.phase = None
            if rt.active and not for_capture and batch.num_reqs == 1:
                slot = int(batch.idx_mapping_np[0])
                encoder = bool(state.diffusion_states.is_encoder_phase[slot].item())
                rt.phase = 'encoder' if encoder or batch.num_draft_tokens == 0 else 'denoise'
                if rt.phase == 'denoise':
                    rt.step += 1
            return original_prepare(state, batch, mode, tables, slots, groups, config,
                                    for_capture=for_capture, ubatch_idx=ubatch_idx)

        def forward(impl, layer, query, key, value, cache, meta, output, output_scale=None, output_block_scale=None):
            if not (rt.active and rt.phase == 'denoise' and '.layers.29.' in getattr(layer, 'layer_name', '')):
                return original_forward(impl, layer, query, key, value, cache, meta, output, output_scale, output_block_scale)
            n = int(meta.num_actual_tokens)
            length = int(meta.seq_lens[0].item())
            prefix = length - n
            if n != 256 or impl.num_heads != 16 or impl.num_kv_heads != 2:
                raise RuntimeError('unqualified native GLOBAL geometry')
            if prefix != rt.prefix:
                rt.prefix, rt.step, rt.keep = prefix, 1, None
                rt.previous_q, rt.certificates = None, []
                rt.last_winner = rt.margin = rt.stable = None
            q = query[:n]
            kc, vc = cache.transpose(1, 2).split(impl.head_size, dim=-1)
            blocks = prefix // 64
            # One dense bootstrap per canvas, and no artificial continuation.
            if rt.mode != 'allkept_layer29' and (rt.step == 1 or blocks == 0):
                rt.previous_q = q.detach().clone()
                rt.eligible += 16 * 2 * ((length + 63) // 64)
                if rt.audit:
                    rt.records.append(dict(step=rt.step, prefix=prefix, phase='native_bootstrap', eligible=16*2*((length+63)//64), skipped=0, local_skipped=0, canvas_skipped=0))
                return original_forward(impl, layer, query, key, value, cache, meta, output, output_scale, output_block_scale)
            refresh = rt.keep is None or (rt.step - 2) % 4 == 0
            needs_kv = refresh or rt.mode == 'validity_reuse_mass_certificate'
            k = v = None
            if needs_kv and rt.mode != 'allkept_layer29':
                page = kc.shape[1]
                pages = meta.block_table[0, :(length + page - 1) // page].long()
                # Materialized CURRENT native cache is temporary selector input.
                k = kc.index_select(0, pages).flatten(0, 1)[:length]
                v = vc.index_select(0, pages).flatten(0, 1)[:length]
            if rt.mode == 'validity_reuse_mass_certificate' and not refresh:
                safe = True
                for unit, cert in enumerate(rt.certificates):
                    if cert['allkept']:
                        continue
                    h, qb = divmod(unit, 2)
                    qu = q[qb*128:(qb+1)*128, h].float()
                    tail = k[blocks*64:, h//8].float()
                    tail_logZ = (qu @ tail.T * impl.scale).logsumexp(-1)
                    lower = mass_lower_bound(cert['anchor_q'], qu, cert['max_prefix_key_norm'],
                        cert['prefix_logZ'], cert['kept_prefix_logZ'], tail_logZ, impl.scale)
                    safe &= bool((lower >= .8).all())
                if not safe:
                    rt.certificate_failures += 1
                    refresh = True
            if rt.mode == 'allkept_layer29':
                rt.keep = torch.ones((1, 16, 2, (length+63)//64), device=q.device, dtype=torch.bool)
                phase = 'allkept'
            elif refresh:
                pair = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if rt.audit else None
                if pair:
                    pair[0].record()
                keep = torch.ones((1, 16, 2, (length+63)//64), device=q.device, dtype=torch.bool)
                certs, banks = [], []
                for h in range(16):
                    for qb in range(2):
                        lo, hi = qb*128, (qb+1)*128
                        previous = rt.previous_q[lo:hi, h] if rt.previous_q is not None else None
                        margin = rt.margin[lo:hi] if rt.margin is not None else None
                        stable = rt.stable[lo:hi] if rt.stable is not None else None
                        support, bank, certificate = choose_unit(q[lo:hi, h], k[:, h//8], v[:, h//8],
                            impl.scale, blocks, previous, margin, stable, rt.mode, h*2+qb)
                        keep[0, h, qb, :blocks] = support
                        rt.fallback_units += bank['chosen'] is None
                        certs.append(certificate)
                        if rt.audit and len(rt.banks) < 2:
                            banks.append(dict(head=h, query_block=qb, **bank))
                rt.keep, rt.certificates = keep, certs
                rt.refreshes += 1
                if banks:
                    rt.banks.append(dict(prefix=prefix, step=rt.step, units=banks))
                phase = 'refresh'
                if pair:
                    pair[1].record(); rt.events.append(('selector_discovery', *pair))
            else:
                rt.held += 1
                phase = 'held'
            rt.previous_q = q.detach().clone()
            from experiments.numerical_qk_reuse import v27_fa4
            from experiments.numerical_qk_reuse.v31_local_sparse import alias64
            kk, vv, table = alias64(kc, vc, meta.block_table[0], length)
            lists = v27_fa4.block_sparse_tensors(rt.keep)
            result = v27_fa4.load()(q[None], kk, vv, softmax_scale=impl.scale, causal=False,
                page_table=table, seqused_k=torch.tensor([length], device=q.device, dtype=torch.int32),
                block_sparse_tensors=lists)[0]
            output[:n].copy_(result[0].reshape_as(output[:n]))
            skipped = int((~rt.keep).sum())
            rt.skipped += skipped; rt.eligible += rt.keep.numel()
            if rt.audit:
                mask_hash = hashlib.sha256(rt.keep.detach().cpu().numpy().tobytes()).hexdigest()
                rt.records.append(dict(step=rt.step, prefix=prefix, phase=phase, eligible=rt.keep.numel(),
                    skipped=skipped, local_skipped=0, canvas_skipped=0, support_sha256=mask_hash,
                    kernel='SM90 FA4 paged sparse consumer with current QKV'))
            return output

        dg.DiffusionGemmaModelState.prepare_attn = prepare
        fa.FlashAttentionImpl.forward = forward
