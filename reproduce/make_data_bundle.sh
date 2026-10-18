#!/usr/bin/env bash
# Carve the publishable reproduction bundle out of a full data root.
#
# Everything in REPRODUCE.md -- Tab. III, Fig. 3, Fig. 4 and Tab. I -- reads
# per-episode OUTCOMES and the rescored ablation sweep. None of it reads video,
# features or checkpoints. So the bundle is ~60 MB carved out of a data root
# that is ~1 TB, and 56 MB of that is the three rescored.json files.
#
#   SRC=/path/to/full/data SRC_STUDY=/path/to/study_results \
#     ./reproduce/make_data_bundle.sh data_bundle
#
# Then check the bundle actually reproduces, which is the point of it:
#
#   DATA_PATH=$PWD/data_bundle STUDY_RESULTS_DIR=$PWD/data_bundle/study_results \
#     ./reproduce/run_all.sh
#
# Tab. II is NOT covered: its benchmark reads policy embeddings and video, which
# are orders of magnitude larger. See docs/progress_estimation.md.
set -euo pipefail

OUT=${1:-data_bundle}
SRC=${SRC:?set SRC to the full data root (the one holding rollouts/ and sweeps/)}
SRC_STUDY=${SRC_STUDY:-}

# The three DROID tasks behind Tab. III. A cell's value depends on the size of
# the nuisance grid behind it, so these are the rescored sweeps, not raw ones.
ABLATION_TASKS=(abc_settable abc_preparetray abc_cooking_v2)

rm -rf "$OUT"
mkdir -p "$OUT/rollouts" "$OUT/sweeps"

echo "== rollouts: per-episode outcomes only =="
# `episodes/metadata.yaml` carries the success flag and length of every episode.
# That is the whole input to the policy success rates and to the study's regret
# read-out; the media and policy_embeddings beside it are not read.
n_coll=0
while IFS= read -r rel; do
    mkdir -p "$OUT/rollouts/$(dirname "$rel")"
    cp "$SRC/rollouts/$rel" "$OUT/rollouts/$rel"
    n_coll=$((n_coll + 1))
done < <(cd "$SRC/rollouts" && find . -maxdepth 3 -name metadata.yaml -printf '%P\n' | sort)
echo "   $n_coll collections"

echo "== sweeps: the rescored ablation =="
for t in "${ABLATION_TASKS[@]}"; do
    src="$SRC/sweeps/$t/rescored.json"
    [ -f "$src" ] || { echo "   MISSING $src" >&2; exit 1; }
    mkdir -p "$OUT/sweeps/$t"
    cp "$src" "$OUT/sweeps/$t/"
    echo "   $t"
done

if [ -n "$SRC_STUDY" ]; then
    echo "== study responses =="
    mkdir -p "$OUT/study_results"
    # Participant response records only. `raffle_entries.jsonl` sits in the same
    # directory and holds participant EMAIL ADDRESSES -- it is never part of a
    # release, and copying by explicit glob rather than by directory is what
    # keeps it out.
    n_resp=0
    for f in "$SRC_STUDY"/*.json; do
        [ -e "$f" ] || continue
        cp "$f" "$OUT/study_results/"
        n_resp=$((n_resp + 1))
    done
    echo "   $n_resp response files (raffle_entries.jsonl deliberately excluded)"
    if compgen -G "$OUT/study_results/*raffle*" > /dev/null; then
        echo "   REFUSING: a raffle file reached the bundle" >&2
        exit 1
    fi
    # Name-matching only catches the file we know about. The responses carry no
    # contact details at all, so anything email-shaped in the copied data means
    # something new arrived in the store and nobody told this script.
    if grep -rlEo '"[^"]+@[^"]+\.[a-z]{2,}"' "$OUT/study_results" 2>/dev/null | head -1 | grep -q .; then
        echo "   REFUSING: an email address reached the bundle -- check $SRC_STUDY" >&2
        exit 1
    fi
fi

{
    echo "Behavior graphs -- reproduction data bundle"
    echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ) from $SRC"
    echo
    echo "rollouts/<collection>/episodes/metadata.yaml   $n_coll files"
    echo "    per-episode success flags and lengths. Inputs to Fig. 3 (policy"
    echo "    success rates) and to the regret read-out behind Fig. 4/Tab. I."
    echo
    echo "sweeps/<task>/rescored.json                    ${#ABLATION_TASKS[@]} files"
    echo "    the rescored representation x reduction sweep behind Tab. III,"
    echo "    for ${ABLATION_TASKS[*]}."
    [ -n "$SRC_STUDY" ] && {
        echo
        echo "study_results/*.json                           ${n_resp:-0} files"
        echo "    participant response records. No contact details: the raffle"
        echo "    entries that live beside them in the working store are excluded."
    }
    echo
    echo "NOT included: video, policy embeddings, checkpoints, and the"
    echo "{dataset}_metrics.json files behind Tab. II."
    echo
    du -sh "$OUT"
} > "$OUT/MANIFEST.txt"

echo
cat "$OUT/MANIFEST.txt"
