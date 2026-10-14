"""Every app entrypoint resolves what it registers.

`apps/demo/Home.py` lists its pages by path and hands them to `st.navigation`,
which raises on a path that does not exist — so a page file removed without
editing the registry takes the whole app down at startup, on every page, for
everyone. Nothing else catches it: `tests/test_demo_app.py` opens
`_pages/graph_demo.py` directly rather than through `Home.py`, which is exactly
the path that skips the registry.

The same holds for the config files the UI reads by path rather than through
hydra (the VLM model list), and for the study app's variant renderers: a yaml
with no renderer is skipped by design, but a renderer whose module is gone is an
ImportError at the first participant.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
APPS = REPO / "apps"


def _string_literals(path: Path) -> list[str]:
    """Every string constant in a module, without importing it."""
    tree = ast.parse(path.read_text())
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


@pytest.mark.parametrize("home", sorted(APPS.glob("*/Home.py")), ids=lambda p: p.parent.name)
def test_every_registered_page_file_exists(home: Path) -> None:
    """A `_pages/<name>.py` named in an entrypoint has to be on disk."""
    named = [s for s in _string_literals(home) if s.endswith(".py")]
    missing = [s for s in named if not (home.parent / "_pages" / s).exists()]
    assert not missing, f"{home.relative_to(REPO)} registers missing pages: {missing}"


def test_the_demo_registers_every_page_it_ships() -> None:
    """And the other way round: a page on disk that nothing registers is dead."""
    pages_dir = APPS / "demo" / "_pages"
    on_disk = {p.name for p in pages_dir.glob("*.py")}
    registered = {s for s in _string_literals(APPS / "demo" / "Home.py") if s.endswith(".py")}
    assert on_disk == registered, (
        f"unregistered: {sorted(on_disk - registered)}; "
        f"registered but absent: {sorted(registered - on_disk)}"
    )


def test_the_ui_config_files_read_by_path_exist() -> None:
    """Config the UI loads directly, outside hydra's composition."""
    from behavior_graphs.viz.streamlit.models import model_choices

    options, default = model_choices()
    assert options, "configs/vlm/models.yaml resolved to no options"
    assert default in options


def test_every_study_variant_with_a_renderer_imports() -> None:
    """A variant yaml without a renderer is skipped; a broken renderer is not."""
    from behavior_graphs.viz.streamlit.user_studies import registry

    variants = {v.id for v in registry.list_variants()}
    assert variants, "no study variants discovered"
    for variant in registry.list_variants():
        assert callable(variant.render), variant.id
