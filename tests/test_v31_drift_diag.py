"""v31 drift diagnostic: per (head, row) relative query / output change between consecutive GLOBAL calls of a layer in
one canvas, counted at the thresholds; nothing is counted across canvases."""
import torch


def test_drift_counts_known_changes():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='mage', drift_diag=True)
    H, n, D = 2, 200, 16
    g = torch.Generator().manual_seed(51)
    q = torch.randn(1, H, n, D, generator=g)
    out = torch.randn(1, n, H, D, generator=g)
    a._drift_account(5, q, out, n)                                     # first call of the canvas: nothing to compare
    assert a._drift_acc is None
    q2 = q.clone()
    q2[0, 0, :10] *= 1.05                                              # 10 rows of head 0 move by 5%
    q2[0, 1, 150] *= 1.5                                               # one row of head 1, block 1, moves by 50%
    a._drift_account(5, q2, out, n)
    r = a._drift_receipt()
    t = r['drift_thresholds']
    i01, i3 = t.index(0.01), t.index(0.3)
    total_rows = H * n
    # bf16 storage of the previous call adds ~0.4% relative error: below 0.01, above 0.001
    assert abs(r['drift_q_rows'][i01] - (total_rows - 11) / total_rows) < 1e-9
    assert abs(r['drift_q_rows'][i3] - (total_rows - 1) / total_rows) < 1e-9
    assert r['drift_q_blocks'][i01] == 2 / 4                           # head 0 block 0 and head 1 block 1 moved
    assert r['drift_out_rows'][i01] == 1.0
    assert r['drift_pairs'] == total_rows
    a.canvas_id += 1                                                   # a new canvas starts a new chain
    a._drift_account(5, q, out, n)
    assert a._drift_receipt()['drift_pairs'] == total_rows
