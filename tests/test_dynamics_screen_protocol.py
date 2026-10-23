import torch

from experiments.diffusion_gemma_sparse_attention_research.dynamics_screen import _select, stable_key
from experiments.diffusion_gemma_sparse_attention_research.native_observer import NativeTrajectoryObserver, _stats


def test_selection_is_score_blind_and_stable():
    rows = [{"source_id": str(i), "score": i} for i in range(20)]
    first = _select(rows, 7, "toy", "source_id")
    second = _select(list(reversed(rows)), 7, "toy", "source_id")
    assert [row["source_id"] for row in first] == [row["source_id"] for row in second]
    assert [row["score"] for row in first] != list(range(7))
    assert stable_key("toy", "0") != stable_key("toy", "1")


def test_native_logit_stats_are_numeric_and_read_only():
    logits = torch.tensor([[[5.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])
    before = logits.clone()
    stats = _stats(logits, 0.005)
    assert torch.equal(logits, before)
    assert stats["finite_logits"]
    assert 0.0 <= stats["confident_position_fraction"] <= 1.0
    assert stats["margin_p50"] >= 0.0


def test_observer_tracks_canvas_boundary_and_stopping_predicate():
    observer = NativeTrajectoryObserver(max_denoising_steps=3, stability_threshold=1)
    observer.begin_request("toy")
    current = torch.tensor([[1, 2]])
    sampled = torch.tensor([[1, 3]])
    logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0]]])
    accepted = torch.tensor([[True, False]])
    observer._new_canvas_if_needed(3)
    observer._record_accept(current, sampled, logits, 3, accepted)
    first_result = (torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), logits, torch.tensor([False]))
    observer._record_after_step(torch.tensor([[0, 0]]), first_result, torch.tensor([False]))
    observer._new_canvas_if_needed(2)
    observer._record_accept(current, sampled, logits, 2, accepted)
    second_result = (torch.tensor([[1, 0]]), torch.tensor([[1, 0]]), logits, torch.tensor([True]))
    observer._record_after_step(torch.tensor([[1, 0]]), second_result, torch.tensor([False]))
    assert [row["canvas_id"] for row in observer.records] == [0, 0]
    assert observer.records[-1]["native_stop_stable"]
    assert observer.records[-1]["native_stop_both"]
    assert observer.summary()["denoising_calls"] == 2
