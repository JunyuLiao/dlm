"""CPU checks for scripts/v26_clock_opportunity.py's reference-clock
reconstruction (no GPU, no model). Run with:
    PYTHONPATH=. python tests/test_v26_clock_opportunity.py
"""
import sys

sys.path.insert(0, ".")

from scripts.v26_clock_opportunity import (  # noqa: E402
    classify_canvas, floor_crosscheck, _is_recorded_policy,
)

FAILURES = []


def check(name, cond):
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}")
        FAILURES.append(name)


def phase_string(n, a_period, r):
    """Render the per-call phase label string for a canvas of length n,
    call-by-call, for human-readable comparison against known schedules."""
    if n <= 0:
        return ""
    labels = ["B0"]
    if n == 1:
        return ",".join(labels)
    labels.append("BO")
    c = classify_canvas(n, a_period, r)
    anchor_set = set(c["anchor_calls"])
    d_set = set(c["d_calls"])
    for call in range(2, n):
        if call in anchor_set:
            labels.append("A")
        elif call in d_set:
            labels.append("D")
        else:
            labels.append("H")
    return ",".join(labels)


# ----------------------------------------------------------------------
# Schedule for n in {1,2,3,9,10,17,18} for R1/R3/R6/B(matched) and A8/A16
# ----------------------------------------------------------------------

def test_schedules():
    ns = [1, 2, 3, 9, 10, 17, 18]

    # n=1: only B0, regardless of policy.
    for a_period in (8, 16):
        for r in (1, 3, 6, a_period):
            c = classify_canvas(1, a_period, r)
            check(f"n=1 a={a_period} r={r} is pure B0",
                  c == dict(B0=1, BO=0, A=0, D=0, H=0, anchor_calls=[], d_calls=[]))

    # n=2: B0, BO only -- the map is never used.
    for a_period in (8, 16):
        for r in (1, 3, 6, a_period):
            c = classify_canvas(2, a_period, r)
            check(f"n=2 a={a_period} r={r} is B0+BO only",
                  (c["B0"], c["BO"], c["A"], c["D"], c["H"]) == (1, 1, 0, 0, 0))

    # n=3, A8, R1: call2 is a D (age 1 >= R1).
    c = classify_canvas(3, 8, 1)
    check("n=3 A8 R1 -> call2 is D", (c["B0"], c["BO"], c["A"], c["D"], c["H"]) == (1, 1, 0, 1, 0))

    # n=3, A8, R3 or R6: call2 has age 1 < R -> H.
    for r in (3, 6):
        c = classify_canvas(3, 8, r)
        check(f"n=3 A8 R{r} -> call2 is H",
              (c["B0"], c["BO"], c["A"], c["D"], c["H"]) == (1, 1, 0, 0, 1))

    # n=3, A8 matched-B (R=8): call2 held.
    c = classify_canvas(3, 8, 8)
    check("n=3 A8 matchedB -> call2 is H", (c["B0"], c["BO"], c["A"], c["D"], c["H"]) == (1, 1, 0, 0, 1))

    # n=9: calls 2..8 (7 calls), no anchor yet (call9 would be first anchor
    # but call index 9 requires n>=10). A8 R1: everything from call2 is D.
    c = classify_canvas(9, 8, 1)
    check("n=9 A8 R1: 7 D, 0 A", (c["A"], c["D"], c["H"]) == (0, 7, 0))
    # A8 R3: D at call4,7 (age from reset=1 at call1: 3,6 -> D at 4 and 7); H elsewhere.
    c = classify_canvas(9, 8, 3)
    check("n=9 A8 R3 phase string", phase_string(9, 8, 3) == "B0,BO,H,H,D,H,H,D,H")
    # A8 R6: D at call7 only (age 6).
    c = classify_canvas(9, 8, 6)
    check("n=9 A8 R6 phase string", phase_string(9, 8, 6) == "B0,BO,H,H,H,H,H,D,H")
    # A8 matched-B: all held, no D.
    c = classify_canvas(9, 8, 8)
    check("n=9 A8 matchedB: D=0", c["D"] == 0)
    # A16: no anchor within 9 calls either way; R1 still floods D.
    c = classify_canvas(9, 16, 1)
    check("n=9 A16 R1: 7 D, 0 A", (c["A"], c["D"], c["H"]) == (0, 7, 0))

    # n=10: call9 exists (index 9 < 10) -> first later anchor under A8.
    c = classify_canvas(10, 8, 3)
    check("n=10 A8 R3 has exactly 1 anchor at call9",
          c["A"] == 1 and c["anchor_calls"] == [9])
    check("n=10 A8 R3 total conserves", c["B0"] + c["BO"] + c["A"] + c["D"] + c["H"] == 10)
    # Under A16, call9 is not an anchor (first anchor would be call17).
    c = classify_canvas(10, 16, 3)
    check("n=10 A16 R3 has no anchor yet", c["A"] == 0)

    # n=17: call9 is an anchor (9<17), call17 is NOT reachable (index 17
    # requires n>=18).
    c = classify_canvas(17, 8, 3)
    check("n=17 A8 R3 has exactly 1 anchor (call9 only)",
          c["A"] == 1 and c["anchor_calls"] == [9])
    check("n=17 A8 R3 total conserves", sum(c[k] for k in ("B0", "BO", "A", "D", "H")) == 17)

    # n=18: call9 and call17 both exist (17 < 18) -> 2 anchors under A8.
    c = classify_canvas(18, 8, 3)
    check("n=18 A8 R3 has 2 anchors (call9, call17)",
          c["A"] == 2 and c["anchor_calls"] == [9, 17])
    check("n=18 A8 R3 total conserves", sum(c[k] for k in ("B0", "BO", "A", "D", "H")) == 18)
    # Under A16: only call17 (=1+1*16) is an anchor.
    c = classify_canvas(18, 16, 3)
    check("n=18 A16 R3 has 1 anchor (call17)", c["A"] == 1 and c["anchor_calls"] == [17])

    # Conservation holds for every (n, a_period, r) in the grid.
    ok = True
    for n in ns:
        for a_period in (8, 16):
            for r in (1, 3, 6, a_period):
                c = classify_canvas(n, a_period, r)
                total = c["B0"] + c["BO"] + c["A"] + c["D"] + c["H"]
                if total != n:
                    ok = False
                    print(f"  conservation broke at n={n} a={a_period} r={r}: total={total}")
    check("conservation B0+BO+A+D+H==n over full grid", ok)

    # Matched-B (R==A_period) always has D==0.
    ok = True
    for n in ns + [25, 33, 50]:
        for a_period in (8, 16):
            c = classify_canvas(n, a_period, a_period)
            if c["D"] != 0:
                ok = False
    check("matched-B (R=A_period) always yields D=0", ok)

    # R=1 (M1-style) always has H==0 for n>=2 (every non-anchor call from
    # call2 onward is immediately a D).
    ok = True
    for n in ns + [25, 33, 50]:
        for a_period in (8, 16):
            c = classify_canvas(n, a_period, 1)
            if n >= 2 and c["H"] != 0:
                ok = False
    check("R=1 always yields H=0 for n>=2", ok)


# ----------------------------------------------------------------------
# Floor cross-check
# ----------------------------------------------------------------------

def test_floor_crosscheck():
    ok = True
    for n in range(0, 60):
        for a_period in (8, 16):
            c = classify_canvas(n, a_period, 3)
            expected, matched = floor_crosscheck(n, a_period, c["A"])
            if not matched:
                ok = False
                print(f"  floor mismatch n={n} a_period={a_period}: predicted={c['A']} expected={expected}")
    check("floor cross-check A==max(0,floor((n-2)/A_period)) holds for n in 0..59", ok)

    # Spot values.
    check("floor(n=12,A=8)=1", ((12 - 2) // 8) == 1)
    check("floor(n=17,A=8)=1", ((17 - 2) // 8) == 1)
    check("floor(n=18,A=8)=2", ((18 - 2) // 8) == 2)
    check("floor(n=2,A=8)=0", max(0, (2 - 2) // 8) == 0)


# ----------------------------------------------------------------------
# Mismatch detector: classify_canvas must actually be sensitive to R,
# A_period and n (i.e. a broken/constant implementation would falsely
# "pass" conservation but fail to distinguish policies).
# ----------------------------------------------------------------------

def test_mismatch_detector():
    # Two different R values on the same n must give different D (for an
    # n large enough to exercise the difference).
    c_r1 = classify_canvas(20, 8, 1)
    c_r3 = classify_canvas(20, 8, 3)
    check("R1 vs R3 differ in D count on n=20", c_r1["D"] != c_r3["D"])

    # Two different A_period values on the same n must give different A
    # (and D, since anchors preempt D).
    c_a8 = classify_canvas(40, 8, 3)
    c_a16 = classify_canvas(40, 16, 3)
    check("A8 vs A16 differ in A count on n=40", c_a8["A"] != c_a16["A"])

    # A deliberately wrong prediction must fail the floor cross-check.
    _, ok_good = floor_crosscheck(12, 8, 1)
    _, ok_bad = floor_crosscheck(12, 8, 2)
    check("floor_crosscheck accepts the correct A", ok_good)
    check("floor_crosscheck rejects a wrong A", not ok_bad)


# ----------------------------------------------------------------------
# _is_recorded_policy labeling
# ----------------------------------------------------------------------

def test_is_recorded_policy():
    check("M3 native bootstrap arm recorded only at A8_R3",
          _is_recorded_policy("M3_native_bootstrap2_observe1", "A8_R3")
          and not _is_recorded_policy("M3_native_bootstrap2_observe1", "A16_R3")
          and not _is_recorded_policy("M3_native_bootstrap2_observe1", "A8_R1"))
    check("M1 native bootstrap arm recorded only at A8_R1",
          _is_recorded_policy("M1_native_bootstrap2_observe1", "A8_R1")
          and not _is_recorded_policy("M1_native_bootstrap2_observe1", "A8_R3"))
    check("B native bootstrap arm recorded only at A8_matchedB",
          _is_recorded_policy("B_native_bootstrap2_observe1", "A8_matchedB")
          and not _is_recorded_policy("B_native_bootstrap2_observe1", "A16_matchedB"))
    check("incumbent arm is never 'recorded' under this clock",
          not _is_recorded_policy("M3_R3_A8_incumbent", "A8_R3"))
    check("native_trajectory rows are never 'recorded'",
          not _is_recorded_policy("native_trajectory:D_native", "A8_R3"))


def main():
    test_schedules()
    test_floor_crosscheck()
    test_mismatch_detector()
    test_is_recorded_policy()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
        raise SystemExit(1)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
