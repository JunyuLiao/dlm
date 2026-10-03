"""Public toy tests: no vLLM, torch, CUDA or remote environment required."""
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.v28_jit_receipts import COUNT_FIELDS, jit_receipts


class JitReceiptTests(unittest.TestCase):
    def monitor(self):
        calls = []

        def handle(*, backend: str, event: str, fn_name: str,
                   detail: str | None = None):
            calls.append((backend, event, fn_name, detail))
            return 'original result'

        return SimpleNamespace(_handle_jit_event=handle), calls

    def event(self, monitor, backend='Triton', event='kernel JIT compilation', **kwargs):
        return monitor._handle_jit_event(backend=backend, event=event,
                                         fn_name='public_toy_kernel', **kwargs)

    def test_real_backend_names_and_unknown_events(self):
        monitor, calls = self.monitor()
        with jit_receipts(monitor) as receipt:
            self.event(monitor)
            self.event(monitor, 'CuTeDSL', 'JIT compilation')
            self.event(monitor, 'CuTe', 'JIT compilation')
            self.event(monitor, 'TileLang', 'JIT compilation')
            self.event(monitor, 'Triton', 'future event')
            self.assertEqual(receipt.snapshot(), dict(triton_events=1, cute_events=1,
                                                     unknown_events=3, all_events=5))
        self.assertEqual(len(calls), 5)

    def test_non_string_unhashable_parameter_delegates_as_unknown(self):
        monitor, calls = self.monitor()
        with jit_receipts(monitor) as receipt:
            self.event(monitor, [], {})
            self.assertEqual(receipt.snapshot()['unknown_events'], 1)
        self.assertEqual(calls[0][:2], ([], {}))

    def test_arguments_and_return_value_preserved(self):
        monitor, calls = self.monitor()
        detail = object()
        with jit_receipts(monitor):
            self.assertEqual(self.event(monitor, detail=detail), 'original result')
        self.assertIs(calls[0][3], detail)

    def test_omitted_default_not_injected_into_delegation(self):
        captured = []

        def original(*, backend, event, fn_name, detail=None):
            pass

        # A wrapper may distinguish omitted arguments while advertising the
        # same real signature; the receipt helper must not rewrite the call.
        def handle(*args, **kwargs):
            captured.append(kwargs.copy())
            return original(*args, **kwargs)

        handle.__signature__ = inspect.signature(original)
        monitor = SimpleNamespace(_handle_jit_event=handle)
        with jit_receipts(monitor):
            self.event(monitor)
        self.assertNotIn('detail', captured[0])

    def test_signature_and_exact_handler_restored(self):
        monitor, _ = self.monitor()
        original = monitor._handle_jit_event
        with jit_receipts(monitor):
            self.assertEqual(inspect.signature(monitor._handle_jit_event), inspect.signature(original))
            self.assertIs(monitor._handle_jit_event.__wrapped__, original)
        self.assertIs(monitor._handle_jit_event, original)

    def test_monitor_error_still_raised_and_counted(self):
        failure = RuntimeError('toy monitor error mode')

        def handle(*, backend, event, fn_name, detail=None):
            raise failure

        monitor = SimpleNamespace(_handle_jit_event=handle)
        with self.assertRaises(RuntimeError) as caught:
            with jit_receipts(monitor) as receipt:
                self.event(monitor)
        self.assertIs(caught.exception, failure)
        self.assertIs(monitor._handle_jit_event, handle)
        self.assertEqual(receipt.snapshot()['triton_events'], 1)

    def test_body_exception_also_restores(self):
        monitor, _ = self.monitor()
        original = monitor._handle_jit_event
        with self.assertRaisesRegex(ValueError, 'toy body'):
            with jit_receipts(monitor):
                raise ValueError('toy body')
        self.assertIs(monitor._handle_jit_event, original)

    def test_invalid_calls_preserve_original_errors_without_events(self):
        monitor, _ = self.monitor()
        original = monitor._handle_jit_event
        bad_calls = [(('Triton',), {}), ((), dict(backend='Triton')),
                     ((), dict(backend='Triton', event='kernel JIT compilation',
                               fn_name='toy', unknown='toy'))]
        with jit_receipts(monitor) as receipt:
            for args, kwargs in bad_calls:
                with self.subTest(args=args, kwargs=kwargs):
                    with self.assertRaises(TypeError) as expected:
                        original(*args, **kwargs)
                    with self.assertRaises(TypeError) as actual:
                        monitor._handle_jit_event(*args, **kwargs)
                    self.assertEqual(str(actual.exception), str(expected.exception))
            self.assertEqual(receipt.snapshot()['all_events'], 0)

    def test_snapshots_are_copies_and_request_deltas_exclude_warm(self):
        monitor, _ = self.monitor()
        with jit_receipts(monitor) as receipt:
            self.event(monitor)
            before = receipt.snapshot()
            self.event(monitor, 'CuTeDSL', 'JIT compilation')
            after = receipt.snapshot()
            self.assertEqual({k: after[k] - before[k] for k in COUNT_FIELDS},
                             dict(triton_events=0, cute_events=1, unknown_events=0, all_events=1))
            before['all_events'] = 999
            self.assertEqual(receipt.snapshot()['all_events'], 2)
            self.assertEqual(set(after), set(COUNT_FIELDS))

    def test_log_deduplication_does_not_deduplicate_receipts(self):
        seen = set()

        def handle(*, backend, event, fn_name, detail=None):
            seen.add((backend, event, fn_name))

        monitor = SimpleNamespace(_handle_jit_event=handle)
        with jit_receipts(monitor) as receipt:
            for _ in range(4):
                self.event(monitor)
            self.assertEqual(receipt.snapshot()['all_events'], 4)
        self.assertEqual(len(seen), 1)

    def test_multiple_event_threads_and_snapshot_invariant(self):
        monitor, _ = self.monitor()
        with jit_receipts(monitor) as receipt:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda _: self.event(monitor), range(200)))
            snap = receipt.snapshot()
            self.assertEqual(snap['all_events'], 200)
            self.assertEqual(sum(snap[k] for k in COUNT_FIELDS[:-1]), snap['all_events'])

    def test_nested_contexts_restore_in_lifo_order(self):
        monitor, _ = self.monitor()
        original = monitor._handle_jit_event
        with jit_receipts(monitor) as outer:
            outer_handler = monitor._handle_jit_event
            with jit_receipts(monitor) as inner:
                self.event(monitor)
            self.assertIs(monitor._handle_jit_event, outer_handler)
            self.event(monitor)
        self.assertEqual(inner.snapshot()['all_events'], 1)
        self.assertEqual(outer.snapshot()['all_events'], 2)
        self.assertIs(monitor._handle_jit_event, original)

    def test_existing_module_lookup_does_not_import_vllm(self):
        monitor, _ = self.monitor()
        with patch.dict(sys.modules, {'vllm.utils.jit_monitor': monitor}):
            with jit_receipts() as receipt:
                self.event(monitor)
        self.assertEqual(receipt.snapshot()['all_events'], 1)

    def test_missing_or_incompatible_handler_rejected(self):
        for monitor in (SimpleNamespace(), SimpleNamespace(_handle_jit_event=1),
                        SimpleNamespace(_handle_jit_event=lambda value: None)):
            with self.subTest(monitor=monitor):
                with self.assertRaises(RuntimeError):
                    with jit_receipts(monitor):
                        self.fail('must not enter')


if __name__ == '__main__':
    unittest.main()
