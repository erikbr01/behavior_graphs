# The test suite

103 test files, ~1840 tests, in three selectable groups. This page is what an
audit of them concluded — where the time goes, which tests are
worth their cost, and which shapes to stop writing — so the next person does not
have to measure it again.

## Running it

Three groups. Two are opt-in, so `pytest` on its own is the fast loop.

```bash
pytest                  # ~1790 tests, ~24 s
pytest -m integration   # trains a cheap lift/square policy, needs GPU + data
pytest -m browser       # 26 tests, ~2 min — headless Chrome over CDP

pytest -m 'not integration and not browser'   # everything but the simulators
```

`-n 8 --dist loadfile` is in `addopts`, so every group runs parallel. Nothing
here shares mutable state between tests — every fixture is `tmp_path` or
in-process — and `loadfile` keeps a file on one worker so its module-scoped
fixtures are paid once rather than once per worker.

`--dist load` was tried and is slower, so `loadfile` stays. Back to back on the
same 32-core box, a bare `pytest` is **42 s** under `loadfile` and **69 s** under
`load`. `load` scatters tests round-robin rather than by file, so per-process
warm-up is paid on more workers; the warning count going 12 → 14 is that
duplication made visible — the umap and numba-TBB init warnings appear once per
worker that lands a UMAP test. The effect is not from scoped fixtures: at `-n 8`
over just the six files with `module`/`class` fixtures, the two modes are within
5–25 % of each other.

`-m` on the command line **replaces** the `-m` in `addopts` rather than adding
to it. That is what makes `-m integration` and `-m browser` work at all, and it
is also why the last line has to spell out the exclusion it wants.

## The trap: the default config makes every pipeline test fit a UMAP

`configs/config.yaml` ships `reduction: umap` and `viz_reduction: umap_2d`. So
**any test that composes a pipeline config and does not override them fits a
UMAP**, whether or not the reduction is what it is testing — and UMAP's numba
JIT is ~20 s per process, once, charged to whichever test gets there first.

`run_pipeline` gives you the escape for free: when the reduced space is already
two wide it reuses `reduced` for the 2-D coords instead of fitting the viz
reducer (`pipeline/run.py:207`). `test_cache.py` had this right all along —
it pins `pipeline/reduction@reduction=pca` and `reduction.n_components=2`, and
the whole file runs in 3 s.

Applying the same to the tests where the reduction was incidental, measured
serially per file:

| | before | after |
|---|---|---|
| `test_sweep.py` | 29.2 s | **4.1 s** |
| `test_run_sweep.py` | 14.0 s | **2.2 s** |
| `test_terminal_subclusters.py` | 12.9 s | **1.4 s** |

None of those tests assert anything about a reduction or about the 2-D coords —
`test_terminal_subclusters` compares an overlay-on run against an overlay-off
one, and `test_run_sweep` checks the script writes its spec and index. They were paying 10x their own
cost for a compile they never used. It also removed 15 spurious UMAP `n_jobs`
warnings from the run.

**So: when a test drives the pipeline but is not about the reduction, pin
`reduction=pca` at `n_components=2`** (or pass
`viz/viz_reduction@viz_reduction=pca_2d` if it needs a wider reduced space, as
the sweep tests do). Note that hydra will not parse a `group@key=` override
nested inside a list value, so for the sweep runner set
`cfg.extra_overrides` after composing rather than on the command line.

What is left is where UMAP genuinely is the subject:
`test_stages.py::TestReduction::test_umap_reduction`, and the two
`reduce_for_subclustering` shape tests in `test_subclusters.py` (UMAP is the
default subcluster reduction). Deselecting those as well gave 88/78/92 s
against 136/99/85 s for three interleaved pairs — no signal, so they stay.

## Why the timings here mislead

Read this before trusting any per-test or per-file number, including the ones
below.

**Two large one-time costs get billed to an arbitrary test.** `import torch` is
~4 s, and UMAP's numba JIT is ~25 s. Both are per-process, and pytest charges
them to whichever test reaches them first — which depends entirely on what else
you selected.

- `tests/test_outcome_classifier.py` reaches torch only through
  `pytest.importorskip("torch")` *inside* test bodies. So
  `TestSafeMlp::test_recovers_a_linearly_separable_boundary` measures **14.4 s**
  when it is the first torch user in the process and **1.8 s** when another test
  imported torch first. Same test, same machine, 8x apart.
- A sweep test that selects a reduction whose output is wider than 2-D is the
  first in its file to fit a viz UMAP and eats the whole JIT: `run_pipeline`
  skips the viz reduction when the reduced space is already 2-D, so a sibling
  test with PCA `n_components=2` takes a second and its neighbour takes thirty.
  Same fixture, same assertions, 30x apart.

## How a run hangs forever, and why it cannot any more

One full-suite run was found parked for **two days** at 0 % CPU: stopped at 82 %
with no output, no failure and no diagnostic, still holding 510 MiB of GPU memory
and 109 threads. It had to be diagnosed by reading `/proc`. The cause is
specific, reproducible, and was in the suite from the moment both halves existed.

**1. Streamlit's `AppTest` repoints `__main__`.** Rendering an app sets
`sys.modules["__main__"].__file__` to the script being rendered, and
`AppTest.from_string` — which most UI tests here use — renders a *temporary*
file that is deleted afterwards. So after any Streamlit test, this process's
`__main__.__file__` names a path that no longer exists:

```
after tests/test_video_browser.py   __main__.__file__ -> '/tmp/tmpj19jvova/1335417...'
```

**2. `multiprocessing`'s `spawn` bootstraps children from exactly that path.**
`test_sweep.py::test_concurrent_sweeps_do_not_lose_each_others_job_keys` opens a
six-worker spawn `Pool` to prove the `flock` in `_merge_jobs` stops a lost
update. Each child re-imports `__main__` by path in
`spawn.py::_fixup_main_from_path`, gets `FileNotFoundError`, and dies at
startup — all six of them.

**3. `Pool` respawns dead workers forever.** Unlike `concurrent.futures`, which
raises `BrokenProcessPool`, `multiprocessing.Pool`'s maintenance thread keeps
replacing dead workers while `map` blocks on a result that can never arrive.
The specimen bore this out exactly: its log had been frozen for two days, its
workers were five hours old, and by the time it was killed they were `<defunct>`
zombies the blocked parent had never reaped.

**4. Nothing bounded it** — no `pytest-timeout`, no `faulthandler_timeout`, and a
bare `pool.map` — so it was silent rather than red.

**Why it looked intermittent.** Serial runs put every test in one process, so
they *always* hit it — which is what main's no-xdist config does, and what the
two-day run was doing. Under `-n auto --dist loadfile` it depends on whether a
Streamlit file and `test_sweep.py` land on the same worker: at `-n 8` and `-n 12`
they did not and the suite was green, at `-n 4` and `-n 6` they did and it
failed. Parallelising the suite accidentally hid this bug most of the time.

**The fix** is in the test: blank `__main__.__spec__` and remove
`__main__.__file__` around the pool, which makes `get_preparation_data` skip the
main fixup entirely (`sys.path` is passed to children separately, so
`_merge_worker` still imports). `__spec__` has to stay *present* — `spawn.py`
reads it unguarded — so it is set to `None` rather than deleted. Plus
`pool.map_async(...).get(timeout=120)` instead of `pool.map`, so any future
variant of this fails in two minutes instead of hanging, and
`faulthandler_timeout = 300`, which dumps every thread's stack after 300 s in one
test. pytest does **not** abort on that (verified: a test sleeping past the
timeout still passes and the run continues), and it survives xdist (verified at
`-n 2`: banner and stacks reach the terminal), so it is a free diagnostic.

Verified in the configurations that broke: serial after
`tests/test_video_browser.py`, serial after three Streamlit files, `-n 4`, and
`-n auto` — the full tier passed in all of them.

**This supersedes an earlier reading of mine.** I first attributed the hang to
the OOM killer taking a pool worker, on the grounds that `/proc/vmstat` shows 20
`oom_kill`s since boot and one run peaks at 21 GiB. That is all true and it is
*not* what happened: the workers die on a stale `__main__` path, deterministically,
with no memory pressure needed. The `oom_kill` count is a real hazard on a shared
box, but it is a separate one.

### What is still unbounded

Audited every wait in `tests/` and `src/` after the above. Bounded now: the one
`pool.map`, the default-tier ffmpeg fixture (`timeout=120`), the `nvidia-smi`
probe (`timeout=30`). Already bounded: `viz/media.py`'s ffmpeg (`timeout=300`)
and the CDP helpers in `tests/browser/cdp.py`.

Deliberately left unbounded: the four `_run()` helpers in `tests/integration/`,
which `subprocess.run(..., check=True)` a training job. A correct cap there is a
policy question — how long is this policy allowed to train — not something to
guess, and `faulthandler_timeout` will at least print where it is stuck. If you
want a cap, put it on `_run` and make it an env var.

### If you find a hung run

`py-spy dump --pid <pid>` is the fastest answer, but needs
`/proc/sys/kernel/yama/ptrace_scope` to be 0 or root — it is 1 here, so it only
works on your own descendants. Without it: `cat /proc/<pid>/wchan` and
`ls /proc/<pid>/task/` (all threads in `futex_wait_queue` at 0 % CPU is a
deadlock, not slow work), `ls -l /proc/<pid>/fd` for the log it was writing and
whether it holds `/dev/nvidia0`, and `pgrep -aP <pid>` for a pool that is still
respawning.

Beware PID ordering when you do this. The PID counter laps its 4194304 ceiling
about every three days at this box's ~17 forks/s, so **a child can have a lower
PID than its parent** and an older process can have a higher PID than a newer
one (`pid 4141598` is 36 days old; `pid 4180774` is 32). A low PID in
`pgrep -aP` output is not evidence that the process is unrelated.

## Where the time goes

The suite used to be dominated by three files whose fixtures trained a torch
model or ran EM over thousands of rows — a sparse autoencoder, a VAE, and an
explicit-duration HSMM — which together were 71% of it and sat behind an opt-in
`fit` marker. Those reducers and clusterers are not part of the method the paper
reports and were removed with it, and the marker went with them.

What is left is collectively cheap: the whole default tier is ~24 s in parallel,
and most files finish in under 5 s. The two shapes that still cost anything are
the per-process one-time costs the previous section describes (UMAP's numba JIT,
`import torch`) and the `AppTest` runs, which boot a Streamlit script per test.

The lesson that outlived the files: **cost control here means watching the
handful of tests that train something, not trimming test count.** Two habits
keep it that way, neither of which gives up a guarantee:

- **Sharing deterministic fits.** `random_state=0` makes a fit reproducible, so
  a reference fit used by four tests is one module-scoped fixture, not four
  trainings.
- **Sizing fixtures to the claim.** Most tests around a fitted model are
  structural (`diag(A) == 0`, pmf rows sum to 1) and hold at any size; only the
  one that genuinely needs a converged fit should pay for one, and its docstring
  should say so with the measured margin.

## What is load-bearing

Most of it. The audit found no smoke-only tests (nothing whose only assertion
is "it did not raise"), no xfailed tests, one skip (the `torchcodec` fast video
decoder is an optional speed dependency, so `test_vision_backbone_features`
skips one case when it is absent), and no tests for deleted
features: every stage with a test file is still selected by a shipped grid in
`configs/grid/`. The parts that earn their keep most clearly:

- **Contract and round-trip tests** — `test_artifacts`, `test_shared_store`,
  `test_cache`, `test_trajectory_dir`, `test_sweep_staleness`. These guard data
  on disk that outlives any one run, where a silent format change is expensive
  to undo. `test_shared_store`'s "a run is readable at every step" and
  "an interrupted delete is finished on the next run" are the kind of thing no
  amount of care substitutes for.
- **The statistics** — `test_policy_comparison`. Reproducing TRI's published
  four-way letter display, Fisher against the SciPy call, Holm never exceeding
  Bonferroni. Cheap, and wrong answers here get published.
- **App state identity** — `test_graph_selection`, `test_tree_fold`,
  `test_demo_app`. The SVG component re-sends its last click on every rerun, so
  "which markers survive a reset" is a real invariant with a real bug behind it.
- **Config instantiation** — `test_configs`. 29 parametrized cases, 11 s, and it
  catches the whole class of "a config group entry stopped matching its
  interface".
- **The browser tier.** `AppTest` does not execute custom components, so
  clicking a node in the SVG graph is only reachable through a real browser.
  Nothing else covers it. The subtask annotation editor is the same story and
  then some — `test_subtask_annotation_browser` drives its buttons, keyboard and
  drag handles and asserts against the JSON each gesture writes, which is how it
  caught the episode picker's remembered widget value outvoting every
  programmatic navigation.

## What is weak, and why

**Behaviour asserted by reading source text.** Seven tests still do some form of

```python
src = inspect.getsource(render._render_submit)
assert 'st.session_state.get(f"{pfx}_remarks", "")' in src
```

This passes whether or not the function works, fails on any rename of a local
variable, and says nothing about what a participant sees. The remaining sites:

| file | line | function it reads |
|---|---|---|
| `test_exit_questions.py` | 66 | `render._render_submit` |
| `test_study_variants.py` | 106 | `contrastive_same_page.render` |
| `test_study_variants.py` | 165 | `video_browser.render_video_browser` |
| `test_study_variants.py` | 212 | `render._render_phase_with_questions` |
| `test_video_browser_options.py` | 71 | `render._render_phase_with_questions` |
| `test_node_panel_descriptions.py` | 74 | two `render.py` files, as text |
| `test_panel_layout.py` | 320 | `trajectory_tree_view._render_native_svg` |

Four of these read `_render_phase_with_questions`, which needs a run dir with
media to render — the reason the shortcut was taken. `tests/test_task_questions.py`
is what the replacement looks like: three source-text assertions over
`render_task_questions` became nine tests that render the block, fill it in, and
check the Continue gate actually opens. Same second of runtime, and they would
survive a rename.

Not in this category, and worth keeping: `test_display_filter`'s "every viz
discovery site uses the filtered helper" and `test_study_variants`'s AST scan
over the variant modules. Those assert an *architectural* rule across files,
which is what static reading is for.

**Assertions that restate a constant.** `assert RATING_STEP == 20` next to
`RATING_STEP = 20` cannot fail for a reason anyone cares about. Free to run, so
not worth a cleanup pass of their own, but not worth writing either. Where a
constant genuinely matters, assert what depends on it: the sibling test that
renders the radio and checks its options are exactly `RATING_POINTS` is the
version with teeth.

**One-shot migration coverage.** `test_import_policy_doctor.py` and
`test_import_policy_doctor_videos.py` (291 lines) cover
`scripts/legacy/import_policy_doctor_*.py`, which imported the pre-repo
clusterings once. They cost ~0.2 s, so there is no cost argument for deleting
them; the argument for deleting them is that they will never fail usefully.
Left in place — whoever decides the legacy importers are gone should take the
tests with them.

## Rules of thumb for new tests

1. **A test that trains a model is a cost centre.** If it costs more than a
   second or two, it belongs in `fit` — which means it does not run by default,
   so say what it buys in the docstring. Give it the smallest fixture the claim
   survives, and record the measured margin so the size is auditable.
2. **A deterministic fit used twice is a fixture.** `scope="module"`, and
   `--dist loadfile` keeps it that way under `-n 8`.
3. **Never assert on source text to check behaviour.** If rendering it is
   awkward, that is a signal about the function, not a licence to read it.
4. **Watch for assertions that cannot fail.** One was found here — a Viterbi
   optimality check of the form `assert score(a) >= score(b) or a != b`, which
   is a tautology. If an assertion has an `or` in it, check the disjunction is
   not always satisfiable, and consider a companion test that the comparison
   can distinguish a known-worse input (`test_the_joint_score_can_tell_a_worse_path_apart`).
