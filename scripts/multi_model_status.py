"""Read-only campaign progress, excluding raw prompts and generations."""
import argparse
import datetime
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    panels = {}
    for path in sorted((root / "runs").glob("*/*/*/shard*/attempt*/records.jsonl")):
        leaf = path.parent
        parts = path.relative_to(root / "runs").parts
        key = "/".join(parts[:3])
        value = panels.setdefault(key, {"complete_shards": 0, "complete_requests": 0, "partial_requests": 0})
        count = sum(1 for line in path.open() if line.strip())
        complete = (leaf / "final_complete.json").exists()
        value["complete_shards"] += int(complete)
        value["complete_requests" if complete else "partial_requests"] += count
        value["last_record_utc"] = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat()
    value = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "panels": panels,
             "complete_requests": sum(x["complete_requests"] for x in panels.values()),
             "complete_panels": sum(x["complete_shards"] == 4 for x in panels.values()),
             "expected_requests": 5110, "expected_panels": 40}
    for name, path in [("worker", root / "private/active_worker.json"),
                       ("health", root / "health/gpu_health.jsonl")]:
        if path.exists():
            value[name] = json.loads(path.read_text().splitlines()[-1])
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
