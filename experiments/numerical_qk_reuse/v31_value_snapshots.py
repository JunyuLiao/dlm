"""Private sampled tensors for a separate offline attention diagnostic.

Copies occur only in explicitly instrumented audit requests. The recorder has
no effect on the selected map, model output, refresh clock or sampler.
"""
from pathlib import Path

import torch


class SnapshotRecorder:
    def __init__(self, directory, request_number):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_number = request_number
        self.last_map = None
        self.age = 0
        self.saved = set()

    def record(self, adapter, layer, q, buffers, scale, prefix, n):
        if layer != 5 or adapter.canvas_id != 0 or n != 256:
            return
        state = adapter.mage_state.get(layer)
        if state is None or state.get('kept') is None or state['canvas'] != adapter.canvas_id:
            return
        selected = state['lists'] is not self.last_map
        self.last_map = state['lists']
        self.age = 0 if selected else self.age+1
        phase = 'initial_or_refresh' if selected else ('held_near' if self.age == 1 else 'held_far' if self.age == 7 else None)
        if phase is None or phase in self.saved:
            return
        self.saved.add(phase)
        # Serialization is private: no prompts, tokens, outputs or host paths in
        # the exported diagnostic report. CPU copies finish before buffers change.
        torch.save(dict(q=q.detach().cpu(), k=buffers['k'].detach().cpu(),
            v=buffers['v'].detach().cpu(), kept=state['kept'].detach().cpu(),
            scale=scale, prefix=prefix, layer=layer, age=self.age,
            phase=phase, selector=adapter.value_selector or 'current_v31_control'),
            self.directory/f'request{self.request_number:03d}_{phase}.pt')
