"""Synthetic CPU provenance tests; no model, torch, or device dependency."""
import unittest

from experiments.numerical_qk_reuse.new_model_events import (
    ActiveToken, IDLMEventClock, LLaDA32EventClock,
)


def rows(start=0, count=32):
    return [ActiveToken(p, p + 100) for p in range(start, start + count)]


def replace_token(active, position, token):
    return [ActiveToken(r.position, token if r.position == position else r.token_id)
            for r in active]


class LLaDA32EventsTest(unittest.TestCase):
    def test_same_layout_clock_does_not_infer_forward_or_commit(self):
        clock = LLaDA32EventClock()
        initial = clock.observe([], rows(), event=0)
        steady = clock.observe([], rows(), event=71)
        self.assertEqual(initial.identities, steady.identities)
        self.assertEqual(steady.event, 71)  # No Gemma48-step clock cap.
        self.assertFalse(steady.requires_refresh)
        self.assertEqual(steady.protected_positions, {0, 31})
        self.assertEqual(steady.prefix_epoch, 0)

    def test_refinement_and_aba_invalidate_token_identity(self):
        clock = LLaDA32EventClock()
        original = rows()
        first = clock.observe([], original, event=0)
        second = clock.observe([], replace_token(original, 8, 777), event=1)
        third = clock.observe([], original, event=2)
        self.assertEqual(second.changed_positions, {8})
        self.assertIn(8, second.protected_positions)
        self.assertNotEqual(first.identities[8], third.identities[8])
        self.assertEqual(first.prefix_epoch, third.prefix_epoch)
        self.assertEqual(third.changed_positions, {8})

    def test_prompt_rows_and_commit_boundary_are_protected(self):
        clock = LLaDA32EventClock()
        prefix = [r.token_id for r in rows()[:10]]
        clock.observe(prefix, rows(), event=0)
        event = clock.observe(prefix, rows(), event=1, boundary_positions=[17])
        self.assertTrue(set(range(11)) <= event.protected_positions)
        self.assertIn(17, event.protected_positions)
        with self.assertRaisesRegex(ValueError, "committed prefix"):
            clock.observe(prefix, replace_token(rows(), 3, 999), event=2)

    def test_native_block_commit_changes_epoch_and_next_window(self):
        clock = LLaDA32EventClock()
        first = clock.observe([], rows(), event=0)
        prefix = [r.token_id for r in rows()]
        committed = clock.observe(prefix, rows(32), event=1)
        self.assertEqual(committed.prefix_epoch, first.prefix_epoch + 1)
        self.assertEqual(committed.new_positions, set(range(32, 64)))
        self.assertEqual(committed.removed_positions, set(range(32)))
        self.assertTrue(committed.requires_refresh)

    def test_changed_boundary_protection_invalidates_layout(self):
        clock = LLaDA32EventClock()
        before = clock.observe([], rows(), event=0)
        after = clock.observe([], rows(), event=1, boundary_positions=[17])
        self.assertTrue(after.requires_refresh)
        self.assertIn(17, after.protected_positions)
        self.assertNotEqual(before.identities, after.identities)

    def test_layout_guards_and_error_leave_clock_unchanged(self):
        clock = LLaDA32EventClock()
        first = clock.observe([], rows(), event=0)
        for bad in [rows(count=31), list(reversed(rows())),
                    rows()[:-1] + [ActiveToken(0, 100)],
                    replace_token(rows(), 8, -1)]:
            with self.assertRaises(ValueError):
                clock.observe([], bad, event=1)
        with self.assertRaises(ValueError):
            clock.observe([], rows(), event=1, boundary_positions=[1000])
        with self.assertRaises(ValueError):
            clock.observe([], rows(), event=1, boundary_positions=[True])
        with self.assertRaises(ValueError):
            clock.observe([], rows(), event=1, rollback=1)
        unchanged = clock.observe([], rows(), event=1)
        self.assertEqual(first.identities, unchanged.identities)


class IDLMEventsTest(unittest.TestCase):
    def test_strided_absolute_positions_not_rows_or_four_token_advances(self):
        clock = IDLMEventClock()
        active = [ActiveToken(p, p + 100) for p in (8, 9, 10, 11, 14, 17, 20)]
        clock.observe(list(range(100, 108)), active, event=0)
        event = clock.observe(list(range(100, 109)), active, event=1)
        self.assertEqual(event.active_positions, (8, 9, 10, 11, 14, 17, 20))
        self.assertEqual(event.prefix_epoch, 2)  # Actual one-token append.
        self.assertTrue(event.requires_refresh)
        self.assertIn(8, event.protected_positions)
        self.assertIn(9, event.protected_positions)

    def test_rejected_draft_rollback_invalidates_unchanged_prefix(self):
        clock = IDLMEventClock()
        active = rows(8, 7)
        prefix = list(range(100, 108))
        before = clock.observe(prefix, active, event=0)
        after = clock.observe(prefix, active, event=1, rollback=True)
        self.assertEqual(after.protected_positions, set(range(8, 15)))
        self.assertEqual(after.invalidated_positions, set(range(8, 15)))
        self.assertEqual(after.invalidation_epoch, 1)
        self.assertNotEqual(before.identities, after.identities)

    def test_prefix_retraction_and_replacement_require_rollback(self):
        clock = IDLMEventClock()
        prefix = list(range(100, 108))
        clock.observe(prefix, rows(8, 7), event=0)
        with self.assertRaisesRegex(ValueError, "rollback"):
            clock.observe(prefix[:4], rows(4, 7), event=1)
        with self.assertRaisesRegex(ValueError, "rollback"):
            clock.observe([999] + prefix[1:], rows(8, 7), event=1)
        rolled = clock.observe(prefix[:4], rows(4, 7), event=1, rollback=True)
        self.assertEqual(rolled.invalidation_epoch, 1)
        self.assertTrue(rolled.requires_refresh)

    def test_departure_reentry_and_layout_reordering_reject_stale_ids(self):
        clock = IDLMEventClock()
        before = clock.observe([], rows(0, 7), event=0)
        shifted = clock.observe([], rows(1, 7), event=1)
        after = clock.observe([], rows(0, 7), event=2)
        self.assertNotEqual(before.identities[0], after.identities[0])
        self.assertEqual(shifted.removed_positions, {0})
        reordered = clock.observe([], list(reversed(rows(0, 7))), event=3)
        self.assertEqual(reordered.new_positions, set())
        self.assertTrue(reordered.requires_refresh)
        self.assertNotEqual(after.identities[0], reordered.identities[-1])

    def test_request_reset_and_pool_slot_reuse_do_not_share_owner(self):
        clock = IDLMEventClock()
        first = clock.observe([], rows(0, 7), event=0)
        other = IDLMEventClock().observe([], rows(0, 7), event=0)
        self.assertNotEqual(first.identities, other.identities)
        clock.reset()
        reset = clock.observe([], rows(0, 7), event=0)
        self.assertNotEqual(first.identities, reset.identities)

    def test_event_and_decode_width_fail_closed(self):
        clock = IDLMEventClock()
        clock.observe([], rows(0, 7), event=3)
        for bad in [3, 2, True, 3.5]:
            with self.assertRaises(ValueError):
                clock.observe([], rows(0, 7), event=bad)
        with self.assertRaisesRegex(ValueError, "prefill unsupported"):
            clock.observe([], rows(0, 12), event=4)
        with self.assertRaises(ValueError):
            IDLMEventClock(gen_block_size=0)


if __name__ == "__main__":
    unittest.main()
