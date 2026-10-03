"""Offline coefficient audit over archived AIME26 trajectories.

This reconstructs only the *expected* ``T_prior``/``T_hybrid``/``T_run``
coefficient from aggregate accepted, renoised, and flip counts in archived
JSON shards. It cannot reconstruct per-query masks or output quality, but it
reveals whether a proposed prior would remain so large that a uniform router
threshold becomes impractical.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _records(root, condition):
    for path in (Path(root) / "final" / condition).glob("*.json"):
        row = json.loads(path.read_text())
        by_canvas = defaultdict(list)
        for step in row.get("step_records", []):
            by_canvas[int(step["canvas_index"])].append(step)
        yield row, by_canvas


def summarize(root, condition, *, beta=3., gamma=.5, tau=2., canvas=256):
    buckets = defaultdict(lambda: dict(n=0, raw=0., prior=0., hybrid=0., run=0.,
                                        renoised=0., flips=0.))
    for _, canvases in _records(root, condition):
        for records in canvases.values():
            records.sort(key=lambda x: int(x["iteration"]))
            trajectory = 1.
            drift = 0.
            temporal = 0.
            stable_run = 0.
            previous_confidence = None
            for step in records:
                index = int(step["iteration"])
                item = buckets[index]
                item["n"] += 1
                renoised = float(step.get("renoised", 0.))
                accepted = float(step.get("accepted", 0.))
                fraction = renoised / max(renoised + accepted, 1.)
                flips = float(step.get("argmax_flips") or 0.) / max(canvas, 1.)
                # Coefficients available at begin(), before this call's
                # logits and sampler outcome are observed.
                prior_hazard = trajectory + (1. - trajectory) * temporal
                hybrid_hazard = 1. - (1. - trajectory) * (1. - temporal) * (1. - drift)
                run_hazard = (pow(2.718281828, -stable_run / tau) +
                              (1. - pow(2.718281828, -stable_run / tau)) * temporal)
                item["raw"] += 1. + beta * temporal
                item["prior"] += 1. + beta * prior_hazard
                item["hybrid"] += 1. + beta * hybrid_hazard
                item["run"] += 1. + beta * run_hazard
                item["renoised"] += fraction
                item["flips"] += flips
                # Aggregate confidence drift is only an approximation: use
                # the change in the mean processed confidence.
                confidence = step.get("confidence_mean")
                if confidence is not None and previous_confidence is not None:
                    drift = gamma * drift + (1. - gamma) * min(
                        1., abs(float(confidence) - previous_confidence))
                if confidence is not None:
                    previous_confidence = float(confidence)
                trajectory = gamma * trajectory + (1. - gamma) * fraction
                stable_run = (stable_run + 1.) * (1. - fraction)
                if step.get("argmax_flips") is not None:
                    stable_run = 0. if flips > 0. else stable_run
                    temporal = gamma * temporal + (1. - gamma) * flips
    rows = []
    for iteration, item in sorted(buckets.items()):
        n = item["n"]
        rows.append(dict(condition=condition, iteration=iteration, n=n,
                         raw_coefficient=item["raw"] / n,
                         prior_coefficient=item["prior"] / n,
                         hybrid_coefficient=item["hybrid"] / n,
                         run_coefficient=item["run"] / n,
                         renoised_fraction=item["renoised"] / n,
                         flip_fraction=item["flips"] / n))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--condition", default="temporal_s50_seed42")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.root, args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["condition", "iteration"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(condition=args.condition, rows=len(rows), output=str(args.output))))


if __name__ == "__main__":
    main()
