"""Private, bounded real-model trajectory capture for offline diagnostics only."""
from pathlib import Path
import torch


class TrajectoryRecorder:
    def __init__(self, directory, request, dataset):
        self.directory = Path(directory)/f'request{request:03d}'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.calls = {}
        self.first_canvas = None
        self.sparse_canvases = set()

    def record(self, adapter, layer, q, buffers, scale, prefix, n):
        if layer not in (5,11,17,23,29):
            raise AssertionError('unexpected trajectory layer')
        if self.first_canvas is None:
            self.first_canvas = adapter.canvas_id
        canvas = adapter.canvas_id-self.first_canvas
        if self.dataset=='aime26' and prefix//64>128:
            if canvas not in self.sparse_canvases and len(self.sparse_canvases)<2:
                self.sparse_canvases.add(canvas)
        if canvas > 1 and canvas not in self.sparse_canvases:
            return
        key = (canvas,layer)
        call = self.calls.get(key,0)
        self.calls[key] = call+1
        # Every call in two canvases; prefix K/V are immutable within each canvas.
        prefix_file = self.directory/f'canvas{canvas:02d}_layer{layer:02d}_prefix.pt'
        if not prefix_file.exists():
            torch.save(dict(k=buffers['k'][:,:,:prefix].detach().cpu(),
                            v=buffers['v'][:,:,:prefix].detach().cpu()), prefix_file)
        st = adapter.mage_state.get(layer)
        used = adapter._stable_used_kept.get(layer)
        following = st.get('kept') if st is not None and st['canvas']==adapter.canvas_id else None
        torch.save(dict(q=q.detach().cpu(),prefix_file=prefix_file.name,
            tail_k=buffers['k'][:,:,prefix:].detach().cpu(),tail_v=buffers['v'][:,:,prefix:].detach().cpu(),
            current=used.detach().cpu() if used is not None else None,
            next=following.detach().cpu() if following is not None else None,
            scale=scale,prefix=prefix,n=n,layer=layer,canvas=canvas,call=call,dataset=self.dataset,
            selection_event=adapter._stable_used_kept.get(layer) is None and following is not None),
            self.directory/f'canvas{canvas:02d}_call{call:02d}_layer{layer:02d}.pt')
