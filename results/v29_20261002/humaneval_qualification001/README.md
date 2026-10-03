# HumanEval qualification001, CPU scoring002

Frozen generation and scorer source54163f4e7. Four fresh engines each completed
one warm and one timed request on the same declared longest input/seed29001.
All4 scored outputs pass the original official tests with unchanged unprivileged
bwrap isolation; public correct/wrong toys and the strict frozen formal-launch
proof guard pass. This only qualifies the pipeline, not task noninferiority or
request-speed gains. Formal generation has not started at publication.

The per-request measurements and full sanitized execution receipts accompany
the aggregate tables. Remove private identities, hardware UUIDs, method/private
hashes, paths, prompts, tokens, gold and completions. The private full proof and
original generation/scoring files remain unchanged and excluded from Git.

Reserved subprocess GPU time672.373426s includes startup/warm/teardown; inner
worker terminal times total663.996s, a different boundary, not another resource
cost to add. Failed dlm2 CPU scoring had no usable unprivileged OS sandbox and
stopped without weakening isolation (0GPU). Under the user's explicit own-host
transfer authorization, pinned gold was copied read-only into a new own mpk CPU
scorer directory; original bytes were preserved. CPU scoring succeeded there.
A separate public renderer initially failed on an integer dictionary key; only
that exporter was fixed. The successful strict proof/scorer/generation were not
modified, and no GPU request was repeated for the renderer failure.
