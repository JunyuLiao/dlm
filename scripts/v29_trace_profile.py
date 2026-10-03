"""Opt-in private Chrome trace for a new independent frozen diagnostic only.

This wrapper leaves the old profiler/deployments unchanged. No automatic trace
publication or sanitization is performed here. Trace data is always private.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


@dataclass(frozen=True)
class TracePlan:
    directory: Path
    record_shapes: bool

    @property
    def output(self):
        return self.directory / 'chrome_trace.private.json'


def validate_trace_spec(spec, cli_directory, cli_shapes, run_directory, ordinal=1):
    config = spec.get('chrome_trace_export', {})
    if not isinstance(config, dict):
        raise ValueError('trace settings must be frozen as an object')
    enabled, shapes = config.get('enabled', False), config.get('record_shapes', False)
    stack = config.get('with_stack', False)
    if any(type(x) is not bool for x in (enabled, shapes, stack)) or stack:
        raise ValueError('trace booleans must be exact; stack capture is unsupported')
    if shapes != cli_shapes or enabled != (cli_directory is not None):
        raise ValueError('trace CLI differs from frozen diagnostic spec')
    if not enabled:
        if shapes:
            raise ValueError('shape capture needs an enabled trace')
        return None
    if (type(ordinal) is not int or ordinal < 1 or spec.get('profile_ordinal', 1) != ordinal):
        raise ValueError('trace request ordinal differs from frozen warm/selection plan')
    if spec.get('diagnostic_only') is not True:
        raise ValueError('trace requires an independent nonformal diagnostic')
    root = Path(config['own_root']).resolve()
    destination = Path(config['private_directory']).resolve()
    cli = Path(cli_directory).resolve()
    run = Path(run_directory).resolve()
    if destination != cli or not destination.is_relative_to(root) or destination == root:
        raise ValueError('trace destination differs from frozen own directory')
    if destination.is_relative_to(run) or run.is_relative_to(destination):
        raise ValueError('trace output must be separate from the new run directory')
    if destination.exists() or Path(cli_directory).is_symlink() or not destination.parent.is_dir():
        raise ValueError('trace requires a wholly new directory under an existing own parent')
    return TracePlan(destination, shapes)


def validate_bound_tools(binding, tools):
    pinned = {Path(path).resolve() for path in binding['files']}
    if any(Path(tool).resolve() not in pinned for tool in tools):
        raise ValueError('binding must pin both trace tool and original profiler')


def export_no_overwrite(profiler, destination):
    """Export to our temporary file, then install by an exclusive hard link.

    export_chrome_trace overwrites its target, so it may only see our new temp
    file. The final name is never replaced, even if another file appears late.
    Cleanup touches only that allocated temp inode, never an existing output.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('private trace output already exists')
    fd, name = tempfile.mkstemp(prefix='.trace-', suffix='.private.tmp', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    identity = temporary.stat()
    try:
        profiler.export_chrome_trace(str(temporary))
        os.link(temporary, destination)  # Atomic no-replace operation on one filesystem.
    finally:
        try:
            current = temporary.lstat()
            if not temporary.is_symlink() and (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def trace_session_patch(base, plan):
    class PrivateTraceSession(base.ProfileSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace_exported = False

        def request_start(self):
            # Override only the creation flags; selection/ordinal remains original.
            original = self.torch.profiler.profile
            def create(**kwargs):
                kwargs.update(record_shapes=plan.record_shapes, with_stack=False, profile_memory=False)
                return original(**kwargs)
            with patch.object(self.torch.profiler, 'profile', create):
                return super().request_start()

        def stop(self, failed=False, boundary_completed=False):
            super().stop(failed=failed, boundary_completed=boundary_completed)
            if self.finished and boundary_completed and not failed and not self.trace_exported:
                export_no_overwrite(self.profiler, plan.output)
                self.trace_exported = True
                receipt = dict(private=True, diagnostic_only=True, performance_claim_allowed=False,
                               quality_evaluated=False, record_shapes=plan.record_shapes,
                               with_stack=False, profile_memory=False, output_file=plan.output.name,
                               output_bytes=plan.output.stat().st_size, automatically_published=False,
                               caveat='trace may contain private paths/scalar arguments; separate sanitization required')
                with (plan.directory/'trace_receipt.private.json').open('x', encoding='utf8') as f:
                    json.dump(receipt, f, indent=2, allow_nan=False); f.write('\n')
    with patch.object(base, 'ProfileSession', PrivateTraceSession):
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--profile-ordinal', type=int, default=1)
    parser.add_argument('--trace-private-dir', type=Path)
    parser.add_argument('--trace-record-shapes', action='store_true')
    args, rest = parser.parse_known_args(argv)
    from scripts import v29_vllm_cost_profile as base, v27_vllm_panel_run as panel
    binding = panel.read(args.binding); spec = panel.read(binding['spec'])
    panel.validate_binding(binding, spec)
    validate_bound_tools(binding, (Path(__file__), Path(base.__file__)))
    plan = validate_trace_spec(spec, args.trace_private_dir, args.trace_record_shapes, args.run_dir, args.profile_ordinal)
    worker_args = ['--binding', args.binding, '--run-dir', str(args.run_dir),
                   '--summary', str(args.summary), '--profile-ordinal', str(args.profile_ordinal)] + rest
    if plan is None:
        return base.main(worker_args)
    # Must reject old outputs before importing GPU runtime/constructing models.
    if args.run_dir.exists() or args.summary.exists():
        raise ValueError('new diagnostic run directory required')
    plan.directory.mkdir(exist_ok=False)
    # On failure leave this new private directory as evidence; never delete a
    # user file, finished trace, original run or frozen deployment.
    with trace_session_patch(base, plan):
        return base.main(worker_args)


if __name__ == '__main__':
    main()
