"""Where a dataset's annotations live, whatever kind they are.

One rule, shared by every annotation store in this package: annotations sit
*inside* the dataset they describe, under ``annotations/<kind>/``. That is the
form that survives a dataset being copied or moved, so a hand-made label travels
with the episodes it describes, and it is what the deployment does — the demo
service mounts ``$ROLLOUTS_DIR`` read-write for exactly this
(``docker-compose.yml``).

``$BEHAVIOR_GRAPHS_ANNOTATIONS_ROOT`` redirects the tree elsewhere for a dataset
that genuinely cannot be written to: a read-only mount, or footage on a share
somebody else owns. It is the fallback, not the deployment's normal state.

Extracted here because the redirected form has a subtlety that must not fork
between kinds: the directory it lands in is named for the dataset *and*
fingerprinted with its resolved path, so two datasets whose directories share a
basename do not share a store. Both kinds get that from one function, and a new
kind of annotation gets it by asking.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

#: Redirects the whole annotation tree to ``<root>/<dataset>-<digest>/<kind>/``.
#: Off by default: the co-located form is the one that survives a dataset being
#: copied or moved.
ENV_ANNOTATIONS_ROOT = "BEHAVIOR_GRAPHS_ANNOTATIONS_ROOT"

#: The directory annotations live in, inside a rollout dir.
ANNOTATIONS_DIR = "annotations"


def resolved(rollout_dir: Path) -> Path:
    """*rollout_dir* as one canonical path, so two names for it are one dataset.

    Both halves of the redirected directory name are taken from this — a symlink
    and its target must not fork the store, and neither must ``ds/`` reached
    from two different working directories. Falls back to the path as given if it
    cannot be resolved: a dataset on a mount that has gone away still needs a
    stable answer here.
    """
    try:
        return rollout_dir.resolve()
    except OSError:
        return rollout_dir


def store_dir(rollout_dir: Path | str, kind: str) -> Path:
    """Where *rollout_dir*'s annotations of *kind* live.

    Under the rollout dir unless :data:`ENV_ANNOTATIONS_ROOT` redirects it, in
    which case the dataset's directory name — the same key ``configs/data/``
    uses for a dataset — names its subdirectory, with the resolved path
    fingerprinted after it.

    The fingerprint is what keeps two datasets apart when their directories share
    a basename, which the browser explicitly supports: the picker labels entries
    by their path relative to the root "even when two roots hold a dataset of the
    same name", and a sharded collection yields ``ds_a/shards/shard0`` next to
    ``ds_b/shards/shard0``. On the basename alone those shared one store, and
    annotating ep0000 of one silently overwrote ep0000 of the other. The
    co-located form — the default, and the one that survives being moved — is
    already unique by construction and is left alone.
    """
    rollout_dir = Path(rollout_dir)
    root = os.environ.get(ENV_ANNOTATIONS_ROOT, "").strip()
    if root:
        real = resolved(rollout_dir)
        digest = hashlib.sha256(str(real).encode()).hexdigest()[:8]
        return Path(root) / f"{real.name}-{digest}" / kind
    return rollout_dir / ANNOTATIONS_DIR / kind


def unwritable_message(rollout_dir: Path | str, kind: str, error: OSError | None = None) -> str:
    """Why a store could not be written to, and what to do about it.

    Worth a shared function because the answer is a deployment detail the person
    clicking cannot guess. An annotation page fails on the first click of a
    session or not at all, and a bare ``Errno 30`` in a traceback says nothing
    about the mount or the variable that would fix it.

    The diagnosis is keyed on *which* error: a read-only mount, a dataset owned
    by somebody else and a full disk all stop a write, and telling somebody their
    mount is read-only when the disk is full sends them to the wrong file.
    *error* is optional because a delete that reports failure by returning
    ``False`` has no exception to hand over.
    """
    where = store_dir(rollout_dir, kind)
    detail = (error.strerror or str(error)) if error else ""
    opening = f"Could not write to `{where}`" + (f" ({detail})." if detail else ".")
    if os.environ.get(ENV_ANNOTATIONS_ROOT, "").strip():
        return (
            f"{opening} `${ENV_ANNOTATIONS_ROOT}` points there, so that path is what "
            "needs to be writable by the app — on the docker deployment, a "
            "read-write mount owned by `$APP_UID`."
        )
    if error is not None and error.errno == errno.EROFS:
        return (
            f"{opening} Annotations go inside the dataset, and this one is on a "
            "read-only mount: mount it read-write, or set "
            f"`${ENV_ANNOTATIONS_ROOT}` to keep annotations outside it."
        )
    return (
        f"{opening} Annotations go inside the dataset, so that directory has to be "
        "writable by the account the app runs as (`$APP_UID` on the docker "
        f"deployment) — or set `${ENV_ANNOTATIONS_ROOT}` to keep them somewhere "
        "that is."
    )
