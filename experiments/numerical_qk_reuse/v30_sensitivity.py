"""Opt-in causal query-weight ablations on M3, preserving its router/clocks.

`unit_v30` removes query weighting (implicit ones). `confidence_v30` uses the
previous completed call's processed top-1 probability: 1+beta*sqrt(1-p_top).
The latter is Junyu's C prior, not his C_gate or fresh retained-state router.
No accepted-mask reconstruction, sampler patch, support carry or stop change.
"""

MODES = ('unit_v30', 'confidence_v30')


class SensitivityOverride:
    def __init__(self, state, mode):
        if mode not in MODES:
            raise ValueError('Unknown explicitly named V30 sensitivity')
        if (state.method != 'T' or not state.fast_t or state.diagnostics or state.collect_margins
                or state.cgate is not None or getattr(state.router,'density_gate',None) is not None):
            raise ValueError('Override requires the fast-T mainline without C/density gates or diagnostics')
        if getattr(state, '_v30_sensitivity_override', None) is not None:
            raise RuntimeError('Sensitivity already overridden')
        self.state, self.mode = state, mode
        self.original_begin, self.original_observe = state.begin, state.observe_logits
        self.canvas = -1
        self.previous_confidence = None
        self.observed_step = None
        self.calls = dict(begin=0,observe=0,unit_steps=0,confidence_steps=0,protected_first_steps=0,canvas_resets=0)
        state._v30_sensitivity_override = self
        state.begin, state.observe_logits = self.begin, self.observe

    def begin(self, cur_step, canvas):
        import torch
        result = self.original_begin(cur_step, canvas)
        if self.canvas != self.state.canvas:
            self.canvas = self.state.canvas
            self.previous_confidence = None
            self.observed_step = None
            self.calls['canvas_resets'] += 1
        self.calls['begin'] += 1
        if self.mode == 'unit_v30':
            # None is exactly the selector's unweighted path, avoiding a new
            # tensor or host/device operation merely to represent ones.
            chosen = None
            self.calls['unit_steps'] += 1
        elif self.previous_confidence is None:
            chosen = torch.full(canvas.shape[:2],1+self.state.beta,device=canvas.device,dtype=torch.float32)
            self.calls['protected_first_steps'] += 1
        else:
            chosen = (1+self.state.beta*(1-self.previous_confidence).clamp_min(0).sqrt()).clamp(1,1+self.state.beta).contiguous()
            self.calls['confidence_steps'] += 1
        self.state.used_weights = chosen
        self.state.router.query_sensitivity = chosen
        self.state.current['weight_active'] = chosen is not None
        self.state.current['v30_sensitivity'] = self.mode
        return result

    def observe(self, logits, accepted, cur_step):
        if self.state.current is None or self.canvas != self.state.canvas:
            raise RuntimeError('Observe requires a begun canvas step')
        identity = (self.canvas,self.state.iteration)
        if self.observed_step == identity:
            raise RuntimeError('Duplicate completed-step observation')
        if self.state.current['remaining_schedule_step'] != cur_step:
            raise ValueError('Sampler observation belongs to a different step')
        if self.mode == 'confidence_v30':
            import torch
            if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(self.state.used_weights.shape):
                raise ValueError('Processed logits must match current B,Q weights')
            x = logits.float()
            self.previous_confidence = (x.amax(-1)-torch.logsumexp(x,dim=-1)).exp().clamp(0,1).detach()
        # Unit mode needs no argmax/confidence work. Neither variant reads or
        # approximates accepted: vLLM currently passes None for that argument.
        self.observed_step = identity
        self.calls['observe'] += 1

    def receipt(self):
        return dict(mode=self.mode,**self.calls,accepted_mask_used=False,temporal_argmax_observed=False,
                    weighting_state='previous_completed_call_only',router_and_carry_modified=False)

    def close(self):
        self.state.begin,self.state.observe_logits = self.original_begin,self.original_observe
        del self.state._v30_sensitivity_override
