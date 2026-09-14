"""Explicit dispatch/provenance for separately versioned routing refinements."""
import json
from .refinements import CONFIGS, MODES, cache_refined, source_sha
from .run import cache
from .ranking_guards import is_guarded,cache_guarded,source_sha as guard_sha


def is_refined(config):
    return config.get('mode') in MODES


def execution_cache(config):
    if is_guarded(config):return cache_guarded
    return cache_refined if is_refined(config) else cache


def provenance(config):
    if is_guarded(config):return dict(ranking_guard_sha256=guard_sha())
    return dict(refinement_sha256=source_sha()) if is_refined(config) else {}


def require_refinement_smoke(root, fp, configs):
    if any(is_guarded(c) for c in configs):
        smoke=json.loads((root/'ranking_guard_smoke.json').read_text())
        if not (smoke['passed'] and smoke['fingerprint']==fp and smoke['ranking_guard_sha256']==guard_sha()):
            raise RuntimeError('matching signal-only ranking CUDA smoke required')
    if not any(is_refined(c) for c in configs):return
    smoke=json.loads((root/'refinement_smoke.json').read_text())
    if not (smoke['passed'] and smoke['fingerprint']==fp and smoke['refinement_sha256']==source_sha()):
        raise RuntimeError('matching refinement CUDA smoke required')


def assert_policy_provenance(policy, config, fp):
    if policy['fingerprint']!=fp or policy['config']!=config:
        raise RuntimeError('policy operator/config mismatch')
    if policy.get('refinement_sha256')!=provenance(config).get('refinement_sha256'):
        raise RuntimeError('policy refinement source mismatch')
