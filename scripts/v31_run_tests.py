"""Run pytest-style test files where pytest is not installed (the deployment's vLLM env).

Supports the subset the v31 tests use: module-level ``pytestmark`` (skipif), ``pytest.mark.parametrize`` and
``pytest.mark.skipif`` on functions. env V27_ADAPTER_DIR puts an overlay first on the package path.
usage: python v31_run_tests.py TEST_FILE.py [...]; exit code 1 on any failure.
"""
import importlib.util
import os
import sys
import traceback
import types


def _install_stub():
    mod = types.ModuleType('pytest')

    def skipif(cond, reason=''):
        def deco(f):
            if cond:
                f._skip = reason
            return f
        deco._skip_cond = (cond, reason)
        return deco

    def parametrize(names, values):
        def deco(f):
            f._params = getattr(f, '_params', []) + [([n.strip() for n in names.split(',')], list(values))]
            return f
        return deco
    mod.mark = types.SimpleNamespace(skipif=skipif, parametrize=parametrize)
    sys.modules['pytest'] = mod


def main():
    _install_stub()
    if os.environ.get('V27_ADAPTER_DIR'):                     # the overlay holding the files under test comes first
        import experiments.numerical_qk_reuse as pkg
        pkg.__path__.insert(0, os.environ['V27_ADAPTER_DIR'])
    failed = 0
    for path in sys.argv[1:]:
        spec = importlib.util.spec_from_file_location(path.rsplit('/', 1)[-1][:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mark = getattr(m, 'pytestmark', None)
        if mark is not None and getattr(mark, '_skip_cond', (False,))[0]:
            print('SKIP', path, mark._skip_cond[1])
            continue
        for name in sorted(n for n in dir(m) if n.startswith('test_')):
            f = getattr(m, name)
            if getattr(f, '_skip', None):
                print('SKIP', name)
                continue
            groups = getattr(f, '_params', [])
            combos = [{}]
            for names, values in groups:
                combos = [dict(c, **dict(zip(names, v if len(names) > 1 else (v,)))) for c in combos for v in values]
            for kw in combos:
                try:
                    f(**kw)
                    print('PASS', name, kw or '')
                except Exception:
                    failed += 1
                    print('FAIL', name, kw or '')
                    traceback.print_exc(limit=6)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
