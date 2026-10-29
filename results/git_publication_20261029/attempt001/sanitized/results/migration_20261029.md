# Results migration, 2026-10-29

Migrated untracked results from this tree to two new directories on `[private_path_sha256_3bc8c309a69e866e]`: `results_relocation_20261029` and `results_nested_relocation_20261029`. Original paths are absolute symlinks; Git-tracked files and their parent directories remain in place. The nested pass selected untracked directories of at least 256 MiB. A subtree containing relative symlinks was skipped.

Each directory was copied with `rsync -aHAX`, checked with checksum-based `rsync -aHAXcn --delete`, renamed aside, replaced by a symlink, and the source backup removed after cutover. The destination `migration.jsonl` files record individual transfers and skips. There were 185 verified cutovers and no copy or verification failures. The symlinks were checked after completion.

The host mount was writable during migration. Some sandboxed environments expose `[private_path_sha256_313d6724363c0e24]` read-only; those environments can read these paths but require a writable mount for workflows that modify migrated directories. Git and tools that do not follow directory symlinks may enumerate migrated data differently.
