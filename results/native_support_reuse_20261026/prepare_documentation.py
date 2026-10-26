"""Preserve dirty user bodies; prepare clean HEAD plus task-owned documentation."""
from pathlib import Path
import json,hashlib,subprocess
REPO=Path(__file__).resolve().parents[2];S=Path(__file__).resolve().parent;OUT=S/'documentation_updates'
def sha(b):return hashlib.sha256(b).hexdigest()
initial=json.loads((S/'initial_state.json').read_text())
for name,h in initial['protected'].items():assert sha((REPO/name).read_bytes())==h,name
cp=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
prefix=f"""## 2026-10-26 UTC — Controlled support reuse complete; no policy advances

Verified evidence checkpoint `{cp}`. Read
[the review handoff](results/native_support_reuse_20261026/REVIEW_HANDOFF.md),
[method decisions](results/native_support_reuse_20261026/method_decisions.json)
and [identities](results/native_support_reuse_20261026/identities.json).
Native source `8d8e0bf6`, protocol `7267a4ba`; pre-outcome choices `d6761435`;
strict full-off qualification `423db8c4`, protocol `9b00cc93`.

All 528 frozen development treatments finish and score officially on six
questions, seeds 42/43. Nine of 12 initial seed cells have feasible banks;
others retain A. AA/SA/AS/held/refreshed 2/4/matched-change are complete.
All 32 reuse and selector gates fail. Reserved confirmation, clean-online/V31
panel and kernels remain unlaunched for scientific reasons, not resources.

Holding CE10% for four calls improves local/refresh to 0.91494, but local/A is
0.98574 and complete geometric N/A is 1.04443. Response10% held4 has local/refresh
0.92840 and full N/A 1.01969. Smaller attention motion is not a favorable-mediator
certificate. Preserve the five complete-behavior leads: mass10% held2 has
N/A 0.92680, C/A 0.92446, paired N/C 1.00254, observed quality 10/12 like A;
FULL/PW score 9/12. This is fewer canvases, not faster per-canvas convergence.
Six LongBench branches lose correct A answers; no A-wrong answer is rescued.

18 focused tests plus bank parity pass. All 657 official rescores, 1258 applied
operator checks and 57 realized-alias repeat groups pass. 657 native requests;
114207 exact request forwards +1044 outside-execute forwards +30 initialization
dispatches =115281 accounted calls, with startup-classification limits explicit.
5500.413 GPU-worker seconds (1.527893h), five serial workers, zero GPU failures.
H100 idle. One verified waiting CPU supervisor stopped before any model child;
no GPU worker was signaled. No ceiling, unrelated resume or push.

Next: independent review of the mass/margin fewer-canvas cases, CE/sham controls
and all regressions. Do not relax failed gates or launch broader/kernel work.
Validity remains shadow-only. Historical Yuhan reuse, Gaussian32, C-gate, V31/V2
and legacy failures remain distinct; the new reuse investigation was explicitly
authorized without retroactively passing those old gates.

""".encode()
append="""

## 2026-10-26 — Complete controlled selector by reuse lifecycle diagnostic

Evidence: results/native_support_reuse_20261026/REVIEW_HANDOFF.md, identities.json,
method_decisions.json and final_complete.json. Native source8d8e0bf6, protocol
7267a4bacaa6eabb5898b1a90253741bf7307b72200a46175702ae0890ffc6a1; choicesd6761435.
Strict full optional-audit-off source423db8c4, protocol
9b00cc93e2a1e276112e6125aabb8476a84589e165073cea48a02e952337018c.

The newly authorized reuse study completes all528 unique native treatments on
six development question clusters, seeds42/43,1344 canonical policy labels.
Nine of12 initial seed cells have feasible banks; no recruitment or later search.
AA/SA/AS/held2/refresh2/matched-change2/held4/refresh4 are complete with current
QKV, dense LOCAL/protected phases and unchanged native sampler/stopping/RNG/caps.
No restoration, hidden forward, stale values, compensation or cross-canvas carry.

Against A:194fewer/169same/165more local calls, range-11..+7. All target canvases
stop adaptively; complete requests separately contain11 native48-call events.
CE10%held4 local/ref0.91494[0.84524,0.99298], local/A0.98574, fullN/A1.04443.
Response10%held4 local/ref0.92840, local/A1.01348, fullN/A1.01969. All32 reuse
and selector gates fail. First-difference operands agree in278 nonalias cases;
54 satisfy meaningful error/mass matching.180 have smaller held attention motion,
but42of those use more calls. No favorable mediator or full acceleration claim.

Preserve five complete-behavior screens and two explicit fewer-canvas leads:
mass10%held2 and weighted-margin10%held2. Mass2 paired geometric N/A0.92680,
C/A0.92446,N/C1.00254; pooledN1775versusA1792 is a different estimand. Observed
quality10/12matchesA; FULL/PW9/12 already differ without deletion. Readout5%held2
loses a correct LongBench answer. Six distinct treatment branches regress across
two LongBench questions; no A-wrong answer is rescued.91output-capped and92missing
final responses remain in the main528, including87already-wrong AIME30branches.
No population accuracy-noninferiority claim. Original and repaired gates agree.

The old readout ordering still fails CE/direction shams on131matched/93meaningful
observed pairs and faithful original-pool100/67views. Preserve selected-pool bias
and zero imputed outcomes. Prior Response/margin already used fixed-four reuse;
their replication/A1/high-low/restoration failures remain unchanged. Yuhan,
Gaussian32,C-gate,V31,V2,readout,margin and signed-pair identities stay separate.

18focused strict-off tests plus bank parity,657official rescores,1258applied
operator checks and57alias repeat groups pass. Actual/FP32 residual median0.28935;
348/1258exceed0.5and28exceed1. All528first pulses change SC numerically in this
instability cohort. Shadow305/769checks cannot certify safety while actual mass
stays high;43later guard violations do not silently refresh pure held policies.

657native requests:114207exactrequest forwards+1044outside-execute forwards+
30initialdispatches=115281accounted calls.5500.413311GPU-worker seconds(1.527893h)
include startup,warmups,qualification and cleanup; initialization-dispatch limits
are explicit. Five serial GPUworkers finish normally. One verified waiting CPU
supervisor is SIGTERMed before any child, not a GPU worker. Empty-bank CPU reader
failure preserved; qualification003unlaunched,004strict-off passes. No resource
ceiling, paused-campaign resume or push. H100idle; unrelated edits preserved.

Decision: no reserved confirmation, clean online/V31 panel, broad benchmark or
kernel work under this protocol. Initial masks were precomputed; W/S/SN are audit
measures. Independent review may consider a separately frozen fewer-canvas
question. Do not relax these failed local/replication gates. Broader reuse and
other locations are not universally rejected. Keep every mixed/negative result.
""".encode()
ledger="""

## 2026-10-26 — Complete controlled support-reuse development diagnostic

- Canonical: results/native_support_reuse_20261026/REVIEW_HANDOFF.md,
  final_complete.json,identities.json,method_decisions.json and report001/.
  Native8d8e0bf6/protocol7267a4bacaa6eabb5898b1a90253741bf7307b72200a46175702ae0890ffc6a1;
  choicesd6761435; strict qualification423db8c4/protocol9b00cc93e2a1e276112e6125aabb8476a84589e165073cea48a02e952337018c.
-528native treatments,6question clusters,seeds42/43;1344labels,9/12feasible seed
  cells. All pulse/hold/real-refresh/matched-change2/4diagnostics complete.
  657total requests and657official rescores reproduce, with currentQKV/native
  sampler/RNG/forward/scope checks. No arbitrary restoration or cross-canvas carry.
- Local versusA194fewer/169same/165more,alltargetadaptive. CE10%held4 local/ref
  0.91494butfullgeometricN/A1.04443; Response10%held4local/ref0.92840butfullN/A
  1.01969.0/32reuse and selector gates pass; conditional confirmation unlaunched.
- Preserve five full-behavior leads,including mass10%held2andmargin10%held2fewer-
  canvas routes. Mass2N/A0.92680,C/A0.92446,N/C1.00254,quality10/12likeA;
  FULL/PW9/12consumer gain predates deletion. Six correct-to-wrong LongBench
  branches,zero A-wrong rescues. No population noninferiority or clean speedup.
-18focused tests plus bank parity,1258applied checks,57alias repeat groups,
  16model files and source/runtime/scorer/input hashes pass.657requests;
  114207exactrequest forwards+1044outside-execute forwards+30initdispatches=
  115281accounted calls.5500.413311allocated GPU-worker seconds(1.527893h),five
  serial workers,no GPUfailure. Startup classification limits are explicit.
- H100idle. CPUreader/unlaunched qualification preserved; no GPU worker signaled,
  resource ceiling, paused resume or push. Clean online/V31rerun,broad benchmark,
  kernels and new refresh-trigger policies remain unlaunched per scientific gates.
""".encode()
OUT.mkdir(exist_ok=False)
base={n:subprocess.check_output(['git','show','HEAD:'+n],cwd=REPO) for n in ['HANDOFF.md','docs/DECISIONS.md','docs/RESULTS_LEDGER.md']}
assert (REPO/'docs/RESULTS_LEDGER.md').read_bytes()==base['docs/RESULTS_LEDGER.md']
(REPO/'HANDOFF.md').write_bytes(prefix+(REPO/'HANDOFF.md').read_bytes())
(REPO/'docs/DECISIONS.md').write_bytes((REPO/'docs/DECISIONS.md').read_bytes()+append)
(REPO/'docs/RESULTS_LEDGER.md').write_bytes(base['docs/RESULTS_LEDGER.md']+ledger)
for name,value in [('HANDOFF_PREFIX.md',prefix),('DECISIONS_APPEND.md',append),('RESULTS_LEDGER_APPEND.md',ledger)]:
    with (OUT/name).open('xb') as f:f.write(value)
owned={'HANDOFF.md':prefix+base['HANDOFF.md'],'docs/DECISIONS.md':base['docs/DECISIONS.md']+append,'docs/RESULTS_LEDGER.md':base['docs/RESULTS_LEDGER.md']+ledger}
for n,v in owned.items():
    p=S/'private/index_blobs'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
preserved=dict(AGENTS=(REPO/'AGENTS.md').read_bytes()==(S/'private/AGENTS.md').read_bytes(),HANDOFF=(REPO/'HANDOFF.md').read_bytes()[len(prefix):]==(S/'private/HANDOFF.md').read_bytes(),DECISIONS=(REPO/'docs/DECISIONS.md').read_bytes()[:-len(append)]==(S/'private/DECISIONS.md').read_bytes())
assert all(preserved.values())
with (OUT/'preservation.json').open('x') as f:json.dump(dict(passed=True,initial_edits_preserved=preserved,verified_evidence_checkpoint=cp,index_blobs={n:sha(v) for n,v in owned.items()},semantics='Stage HEAD base plus own prefix/append, keep original dirty bodies in worktree'),f,indent=2)
print(json.dumps(preserved))
