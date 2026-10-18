"""Cap the numeric libraries' thread pools, before anything imports them.

Import this first, ahead of numpy/sklearn, in any entry point whose numbers
are meant to be comparable across machines.

The progress benchmark is fully seeded -- `KFoldSplit` depends only on the
success flags, `TunedGraphBackbone` passes `random_state` into K-means, the MLP
takes a seed -- and it still did not reproduce across environments. Measured on
libero_object, same clone, same seed, same two cores:

    metric     published   OMP auto   OMP=1
    voc         0.965915   0.965863   0.965915
    kendall     0.934581   0.934449   0.934581
    mae         0.050836   0.050843   0.050836

Two runs at the same thread count agreed to the last digit; changing the
thread count moved the fourth decimal. K-means computes its distances in
parallel chunks, so the number of threads sets the floating-point summation
order and a few windows near a cluster boundary land in a different cluster.
`mlp` and `auroc` were unaffected, which is what pointed at the clustering.

One thread, so the result depends on the data and the seed rather than on how
many cores the machine happened to offer. The parallelism worth having here is
across datasets (`--shard`), not inside one K-means. `setdefault`, so
`OMP_NUM_THREADS=8 python scripts/... ` still wins for anyone who would rather
have the speed and does not need the comparison.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
