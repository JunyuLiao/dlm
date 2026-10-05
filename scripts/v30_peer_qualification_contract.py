"""Pure guards for the one-shot peer kernel qualification queue."""
from collections import Counter

FORMAL = {
    "longbench": (1888, "54163f4e7c8766aaa4187807420b3cf3167dba81"),
    "aime": (960, "1931db5a6540409946641ebb0b04c06507d48402"),
    "humaneval": (5248, "54163f4e7c8766aaa4187807420b3cf3167dba81"),
}


def strict_formal_ready(status):
    if "failed" in status.get("phase", ""):
        raise RuntimeError("Formal predecessor failed; preserve and review")
    if status.get("complete") is not True:
        return False
    if status.get("phase") != "all_strict_scored_public_exports_ready_for_root_review":
        raise ValueError("Unexpected formal completion state")
    completed = status.get("completed", {})
    if set(completed) != set(FORMAL):
        raise ValueError("All three strict scores required")
    for suite, (count, source) in FORMAL.items():
        row = completed[suite]
        if row.get("requests") != count or row.get("workers") != 32 or row.get("source_commit") != source:
            raise ValueError("Formal score inventory or deployment mismatch")
    return True


def validate_test_receipt(receipt, spec):
    if receipt.get("peer_source_commit") != spec["peer_commit"] or receipt.get("kernel_sha256") != spec["kernel_sha256"]:
        raise ValueError("Qualification source/kernel identity mismatch")
    cases = receipt.get("tests", [])
    if len(cases) != spec["must_pass_cases"] or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Incomplete or duplicate test inventory")
    if Counter(c["name"].split("[", 1)[0] for c in cases) != spec["cases_per_function"]:
        raise ValueError("Wrong test functions or parameter inventory")
    if receipt.get("returncode") != 0 or any(c.get("result") != "passed" for c in cases):
        raise ValueError("Every frozen qualification case must pass; skips are failures")
    return True
