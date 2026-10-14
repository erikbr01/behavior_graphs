"""Statistical separability of the three_piece_d0 curation arms via the Toyota
Research recipe (Statistical Thinking for Robot Policy Evaluation):
pairwise BARNARD'S exact test on pooled success/fail counts, BONFERRONI-corrected
across all k(k-1)/2 comparisons, then a COMPACT LETTER DISPLAY (arms sharing a
letter are NOT statistically separable at the corrected level).

Reads the per-seed result TSVs (mixed formats), pools every eval episode per arm
(each seed = 100 episodes) into a 2x2 for Barnard's, and emits markdown tables
(success mean+/-std, throughput mean+/-std, pooled n, CLD letter). Prints to stdout.
"""

from __future__ import annotations

import argparse
import glob
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import barnard_exact

ROOT = "${DATA_PATH}/training/rgrl/mimicgen_three_piece_curate"
EP_PER_SEED = 100
ALPHA = 0.05

#: (arm, seed) -> (success_rate, mean_steps) results that live only here, not in
#: any TSV under ROOT. They are the original soft-identity seeds (overwritten in
#: id_curation_g1.tsv by the resolver) plus one uniform seed. Kept because
#: dropping them would change every pooled 2x2, but declared in one place and
#: ECHOED at run time -- they are inlined numbers, and a reader comparing this
#: table against the TSVs has to be able to see that.
INLINE_RESULTS = {
    ("gsegzsoftid", 10000): (0.79, 335.4),
    ("gsegzsoftid", 20000): (0.77, 332.6),
    ("gsegzsoftid", 30000): (0.71, 334.9),
    ("gsegzsoftid", 40000): (0.79, 337.0),
    ("uniform", 10000): (0.65, None),
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument(
    "--extra-tsv",
    action="append",
    default=[],
    type=Path,
    help="Additional per-seed result file to ingest. This used to be a hardcoded "
    "path to an ephemeral /tmp agent-task file, read under a bare "
    "`except FileNotFoundError: pass` -- so once that file was reaped every "
    "(arm, seed) it carried vanished silently, changing `arms`, the pooled 2x2s, "
    "len(pairs) and therefore the Bonferroni level, while still printing a "
    "complete-looking CLD table. A path given here MUST exist.",
)
args = ap.parse_args()

succ = defaultdict(dict)  # arm -> {seed: success_rate}
steps = defaultdict(dict)  # arm -> {seed: mean_steps_to_success}


def ingest(line: str) -> None:
    p = line.rstrip("\n").split("\t")
    if len(p) < 3 or not re.match(r"^\d+$", p[1].strip()):
        return
    rest = " ".join(p[2:])
    nums = re.findall(r"[0-9]*\.?[0-9]+", rest)
    if not nums:
        return
    arm, seed = p[0], int(p[1])
    succ[arm][seed] = float(nums[0])
    for x in nums[1:]:
        if float(x) > 5:  # steps (success rate is <=1)
            steps[arm][seed] = float(x)
            break


for f in glob.glob(f"{ROOT}/*.tsv"):
    # skip the older single-eval logs that use a non-per-seed schema
    if any(s in f for s in ("ksweep", "curation_results")):
        continue
    for ln in open(f):
        ingest(ln)
for extra in args.extra_tsv:
    if not extra.is_file():
        # Refuse rather than continue: a missing input changes the correction
        # level, and the table gives no hint that it did.
        raise SystemExit(f"--extra-tsv {extra} does not exist; refusing to run a partial CLD")
    for ln in open(extra):
        ingest(ln)

used_inline = []
for (arm, sd), (rate, stp) in sorted(INLINE_RESULTS.items()):
    if sd not in succ[arm]:
        succ[arm][sd] = rate
        used_inline.append(f"{arm}/seed{sd}={rate}")
    if stp is not None:
        steps[arm].setdefault(sd, stp)
if used_inline:
    print(f"NOTE inlined results not present in any TSV: {', '.join(used_inline)}", file=sys.stderr)

arms = sorted(succ, key=lambda a: -sum(succ[a].values()) / len(succ[a]))
# pooled 2x2 counts
S = {a: sum(round(v * EP_PER_SEED) for v in succ[a].values()) for a in arms}
N = {a: EP_PER_SEED * len(succ[a]) for a in arms}


def mstd(d, a):
    v = list(d[a].values())
    return (sum(v) / len(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)) if v else (None, None, 0)


# pairwise Barnard's exact, Bonferroni corrected
pairs = [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))]
alpha_adj = ALPHA / len(pairs)
sig = set()
pvals = {}
for a, b in pairs:
    tbl = [[S[a], N[a] - S[a]], [S[b], N[b] - S[b]]]
    p = barnard_exact(tbl, alternative="two-sided").pvalue
    pvals[(a, b)] = p
    if p < alpha_adj:
        sig.add(frozenset((a, b)))


def absorb(cols):
    cols = [c for c in cols if c]
    keep = []
    for i, c in enumerate(cols):
        if not any(i != j and c < d or (c == d and j < i) for j, d in enumerate(cols)):
            keep.append(c)
    # dedup
    out = []
    for c in keep:
        if c not in out:
            out.append(c)
    return out


# compact letter display (insert-absorb / multcompLetters)
cols = [set(arms)]
for pr in sig:
    a, b = tuple(pr)
    newcols = []
    for col in cols:
        if a in col and b in col:
            newcols.append(col - {a})
            newcols.append(col - {b})
        else:
            newcols.append(col)
    cols = absorb(newcols)

# order columns by best-ranked (highest-mean) member; assign letters
rank = {a: i for i, a in enumerate(arms)}
cols.sort(key=lambda c: min(rank[x] for x in c))
letters = {a: "" for a in arms}
for li, col in enumerate(cols):
    ch = chr(ord("a") + li)
    for a in col:
        letters[a] += ch

# verify CLD property
for a, b in pairs:
    shared = bool(set(letters[a]) & set(letters[b]))
    issig = frozenset((a, b)) in sig
    assert shared != issig, f"CLD violation {a},{b} sig={issig} shared={shared}"

print(f"Barnard's exact, two-sided; Bonferroni alpha_adj = {ALPHA}/{len(pairs)} = {alpha_adj:.2e}")
print(
    f"pooled episodes per arm = 100 x n_seeds; {len(sig)}/{len(pairs)} pairs statistically separated\n"
)
print("| arm | success mean±std (n) | throughput steps ±std | pooled S/N | CLD |")
print("|---|---|---|---|---|")
for a in arms:
    sm, ss, sn = mstd(succ, a)
    tm, ts, tn = mstd(steps, a)
    tcol = f"{tm:.0f}±{ts:.0f}" if tm else "—"
    print(f"| {a} | {sm:.3f}±{ss:.3f} (n={sn}) | {tcol} | {S[a]}/{N[a]} | **{letters[a]}** |")

print(
    "\nGroups sharing a letter are NOT statistically separable (Barnard's + Bonferroni, alpha=0.05)."
)
# uniform-focused summary
u = "uniform"
sep_from_u = [a for a in arms if a != u and frozenset((a, u)) in sig]
print(
    f"\nStatistically separated FROM uniform ({succ[u] and S[u]}/{N[u]}): "
    + (
        ", ".join(
            f"{a} (p={pvals[tuple(sorted((a, u), key=lambda x: rank[x]))]:.1e})" for a in sep_from_u
        )
        if sep_from_u
        else "NONE"
    )
)
