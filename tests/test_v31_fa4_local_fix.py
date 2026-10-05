"""apply_fa4_local_fix (v31_vllm_paired_bench, FA4_LOCAL_FIX=1) on a stand-in vllm.vllm_flash_attn.cute package (CPU).

The real kernel check is the GPU micro-benchmark (scripts/v31_local_window_bench.py --fix --compare: bitwise equal to the
unpatched call at 16K-131K keys, 0.105 ms flat vs 0.48-3.19 ms). Here: the two edits apply exactly once, only LOCAL
kernels skip the bidirectional full-range reset (producer and consumer alike), the patched copy is imported from disk
under the original module name, interface's class binding follows, a second install is a no-op, and an unexpected
source is refused before anything is replaced."""
import contextlib
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = next(p for p in (_HERE.parent / 'scripts' / 'v31_vllm_paired_bench.py', _HERE / 'v31_vllm_paired_bench.py')
            if p.exists())
spec = importlib.util.spec_from_file_location('v31_vllm_paired_bench_fa4fix', _SRC)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

KERNEL = '''def const_expr(x):
    return x


class FlashAttentionForwardSm90:
    def __init__(self, is_local):
        self.is_local, self._mDynamicCausal = is_local, [0]

    def producer(self, batch_idx=0):
        if True:
            if True:
                if True:
                    if const_expr(self._mDynamicCausal is not None):
                        psc_producer = self._mDynamicCausal[batch_idx]
                        if not psc_producer:
                            return 'full'
                    return 'window'

    def consumer(self):
        if True:
            if const_expr(self._mDynamicCausal is not None):
                # Per-sequence causal: psc == 0 means this sequence is processed
                if not self._mDynamicCausal[0]:
                    return 'full'
            return 'window'
'''


def _vllm_modules():
    return {k: v for k, v in sys.modules.items() if k == 'vllm' or k.startswith('vllm.')}


@contextlib.contextmanager
def fake_vllm():
    """A stand-in vllm.vllm_flash_attn.cute package first on sys.path; any real vllm modules are set aside and restored
    (the host runner's env has vLLM installed)."""
    saved, path0 = _vllm_modules(), list(sys.path)
    for k in saved:
        del sys.modules[k]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cute = tmp / 'pkg' / 'vllm' / 'vllm_flash_attn' / 'cute'
        cute.mkdir(parents=True)
        for d in (cute.parent.parent, cute.parent, cute):
            (d / '__init__.py').write_text('', encoding='utf-8')
        (cute / 'flash_fwd_sm90.py').write_text(KERNEL, encoding='utf-8')
        (cute / 'interface.py').write_text(
            'from vllm.vllm_flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90' + chr(10), encoding='utf-8')
        sys.path.insert(0, str(tmp / 'pkg'))
        importlib.invalidate_caches()
        try:
            yield cute, tmp / 'build'
        finally:
            for k in _vllm_modules():
                del sys.modules[k]
            sys.path[:] = path0
            sys.modules.update(saved)
            importlib.invalidate_caches()


def test_only_local_kernels_keep_the_window_range():
    with fake_vllm() as (cute, build):
        import vllm.vllm_flash_attn.cute.interface as iface
        assert Path(iface.__file__).parent == cute
        orig = iface.FlashAttentionForwardSm90
        assert orig(True).producer() == 'full' and orig(True).consumer() == 'full'       # the bug: LOCAL resets too
        path = B.apply_fa4_local_fix(build_dir=build)
        mod = sys.modules['vllm.vllm_flash_attn.cute.flash_fwd_sm90']
        assert Path(mod.__file__) == Path(path) and Path(path).parent.parent == build and mod._v31_fa4_local_fix
        K = mod.FlashAttentionForwardSm90
        assert iface.FlashAttentionForwardSm90 is K and sys.modules['vllm.vllm_flash_attn.cute'].flash_fwd_sm90 is mod
        assert K(True).producer() == 'window' and K(True).consumer() == 'window'          # LOCAL: window blocks only
        assert K(False).producer() == 'full' and K(False).consumer() == 'full'            # GLOBAL: unchanged
        text = Path(path).read_text(encoding='utf-8')
        assert text.count('is not None and not self.is_local):') == 2
        assert text.replace(' and not self.is_local', '') == KERNEL                       # nothing else changed
        assert (cute / 'flash_fwd_sm90.py').read_text(encoding='utf-8') == KERNEL         # installed file untouched
        assert B.apply_fa4_local_fix(build_dir=build) == path                              # second install: no-op


def test_unexpected_source_is_refused_before_any_swap():
    with fake_vllm() as (cute, build):
        (cute / 'flash_fwd_sm90.py').write_text(KERNEL.replace('# Per-sequence causal', '# changed upstream'),
                                                encoding='utf-8')
        import vllm.vllm_flash_attn.cute.flash_fwd_sm90 as orig
        try:
            B.apply_fa4_local_fix(build_dir=build)
        except AssertionError as e:
            assert 'FA4_LOCAL_FIX' in str(e)
        else:
            raise AssertionError('an unexpected flash_fwd_sm90.py was patched')
        assert sys.modules['vllm.vllm_flash_attn.cute.flash_fwd_sm90'] is orig and not build.exists()
