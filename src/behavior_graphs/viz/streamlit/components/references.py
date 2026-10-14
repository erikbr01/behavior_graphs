"""The works the reviewer page cites, numbered as a paper numbers them.

Ordered by **first appearance on the page**, which is what the bracketed
numbers mean: a reader who meets [4] before [3] is reading a list that has
drifted from the text. :func:`check_order` is the guard, and the site's test
suite runs it against the rendered page rather than against this file.

The page itself is anonymous; these are other people's papers, so they carry
nothing the review has to withhold.
"""

from __future__ import annotations

from collections.abc import Iterable

#: ``key -> reference``, in the order the page first cites them. IEEE style,
#: shortened to the first author where the list runs past three: a numbered
#: reference is a pointer, and the page is not a bibliography.
REFERENCES: tuple[tuple[str, str], ...] = (
    (
        "umap",
        "L. McInnes, J. Healy, N. Saul, and L. Großberger, "
        '"UMAP: Uniform manifold approximation and projection," '
        "Journal of Open Source Software, vol. 3, no. 29, p. 861, 2018.",
    ),
    (
        "kmeans",
        'S. P. Lloyd, "Least squares quantization in PCM," '
        "IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129-137, 1982.",
    ),
    (
        "lbm",
        "J. Barreiros et al., "
        '"A careful examination of large behavior models for multitask '
        'dexterous manipulation," Science Robotics, vol. 11, no. 113, 2026.',
    ),
    (
        "letters",
        'H.-P. Piepho, "An algorithm for a letter-based representation of '
        'all-pairwise comparisons," Journal of Computational and Graphical '
        "Statistics, vol. 13, no. 2, pp. 456-466, 2004.",
    ),
    (
        "comparison",
        "D. Snyder et al., "
        '"Is your imitation learning policy better than mine? Policy '
        'comparison with near-optimal stopping," in Robotics: Science and '
        "Systems (RSS), 2025.",
    ),
    (
        "sarm",
        'Q. Chen et al., "SARM: Stage-aware reward modeling for long horizon '
        'robot manipulation," arXiv:2509.25358, 2025.',
    ),
    (
        "libero",
        'B. Liu et al., "LIBERO: Benchmarking knowledge transfer for lifelong '
        'robot learning," in Advances in Neural Information Processing '
        "Systems, vol. 36, 2023, pp. 44776-44791.",
    ),
    (
        "droid",
        'A. Khazatsky et al., "DROID: A large-scale in-the-wild robot '
        'manipulation dataset," arXiv:2403.12945, 2024.',
    ),
    (
        "robomimic",
        'A. Mandlekar et al., "What matters in learning from offline human '
        'demonstrations for robot manipulation," in Conference on Robot '
        "Learning (CoRL), 2021, pp. 1678-1690.",
    ),
    (
        "mimicgen",
        'A. Mandlekar et al., "MimicGen: A data generation system for scalable '
        'robot learning using human demonstrations," in Conference on Robot '
        "Learning (CoRL), 2023, pp. 1820-1864.",
    ),
    (
        "robometer",
        'A. Liang et al., "Robometer: Scaling general-purpose robotic reward '
        'models via trajectory comparisons," in Robotics: Science and Systems '
        "(RSS), 2026.",
    ),
    (
        "topreward",
        'S. Chen et al., "TOPReward: Token probabilities as hidden zero-shot '
        'rewards for robotics," arXiv:2602.19313, 2026.',
    ),
)

_NUMBERS = {key: index for index, (key, _) in enumerate(REFERENCES, start=1)}


def cite(*keys: str) -> str:
    """The bracketed marker for *keys*, e.g. ``"[3], [4]"``.

    Raises on an unknown key rather than rendering an empty bracket: a citation
    that silently disappears is worse on a reviewed page than a crash in a
    test.
    """
    missing = [key for key in keys if key not in _NUMBERS]
    if missing:
        raise KeyError(f"no reference for {missing}; known: {sorted(_NUMBERS)}")
    return ", ".join(f"[{_NUMBERS[key]}]" for key in keys)


def numbered() -> tuple[tuple[int, str], ...]:
    """``(number, reference)`` for every entry, in order."""
    return tuple((index, text) for index, (_, text) in enumerate(REFERENCES, start=1))


def cited_keys(source: str) -> list[str]:
    """Every key passed to :func:`cite` in *source*, in order of first use."""
    import re

    order: list[str] = []
    for match in re.finditer(r"""cite\(([^)]*)\)""", source):
        for key in re.findall(r"""['"]([^'"]+)['"]""", match.group(1)):
            if key not in order:
                order.append(key)
    return order


def check_order(source: str) -> None:
    """Raise unless *source* cites the references in this module's order.

    Checked against the **page source** rather than the rendered page: a
    citation reaches a reader in the order the script writes it, and the
    rendered blocks come back grouped by element type, which is a different
    order and not the one a number has to follow.
    """
    used = cited_keys(source)
    expected = [key for key, _ in REFERENCES if key in used]
    if used != expected:
        raise AssertionError(f"citations appear out of order: {used} (expected: {expected})")
    uncited = [key for key, _ in REFERENCES if key not in used]
    if uncited:
        raise AssertionError(f"references never cited: {uncited}")


def as_markdown(entries: Iterable[tuple[int, str]] | None = None) -> str:
    """The reference list, one numbered line per entry."""
    rows = numbered() if entries is None else tuple(entries)
    return "\n".join(f"{number}. {text}" for number, text in rows)
