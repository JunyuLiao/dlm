import copy

import pytest

from scripts.v31_value_publication_20261017 import check_arm, labels_for


def fixture():
    labels = labels_for("current_v31_control")
    records = {}
    scores = {}
    for i, label in enumerate(labels):
        records[label] = [{"denoise_forwards": i + 1, "canvases": 1,
                           "output_tokens": 10, "cuda_graph_captures": 0,
                           "receipts": {"adapter": {"global_eligible_tiles": 10,
                               "global_kept_tiles": 5, "overall_eligible_rectangles": 20,
                               "overall_skipped_rectangles": 5, "native_local_skipped_tiles": 0}}}]
        scores[label] = {"cell": i % 2}
    result = {"problem_trials": 4, "score_points": 50,
              "work": {"N": 10, "mean_N": 2.5, "C": 4, "mean_C": 1,
                       "T": 40, "mean_T": 10, "global_eligible_tiles": 40,
                       "global_kept_tiles": 20, "overall_eligible_rectangles": 80,
                       "overall_skipped_rectangles": 20}}
    return result, labels, records, scores


def test_control_has_four_bookends_and_candidate_has_two_trials():
    result, labels, records, scores = fixture()
    assert len(labels) == 4
    assert len(labels_for("full_swap8")) == 2
    check_arm(result, labels, records, scores, 100)
    bad = copy.deepcopy(result)
    bad["problem_trials"] = 2
    with pytest.raises(RuntimeError, match="repetition count"):
        check_arm(bad, labels, records, scores, 100)


def test_wrong_local_field_cannot_silently_default_to_zero():
    result, labels, records, scores = fixture()
    del records[labels[0]][0]["receipts"]["adapter"]["native_local_skipped_tiles"]
    with pytest.raises(KeyError, match="native_local_skipped_tiles"):
        check_arm(result, labels, records, scores, 100)


def test_changed_score_or_work_fails_public_audit():
    result, labels, records, scores = fixture()
    scores[labels[0]]["cell"] = 1
    with pytest.raises(RuntimeError, match="score arithmetic"):
        check_arm(result, labels, records, scores, 100)
    result, labels, records, scores = fixture()
    result["work"]["global_kept_tiles"] += 1
    with pytest.raises(RuntimeError, match="physical work"):
        check_arm(result, labels, records, scores, 100)
