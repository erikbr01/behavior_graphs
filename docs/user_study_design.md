# User study design

Recommendations for running the behavior-graph user study across three tasks
so that it produces a defensible result at a sample size we can actually
recruit. Covers the measured ground truth, what's already built, the design
to run, the question battery, and the work that gates it.

The study tests two claims:

- **A — interpretability.** The behavior graph makes a policy's failure
  structure legible.
- **B — decision support.** The behavior graph leads to better
  data-collection decisions.

Both are now measurable rather than self-reported, because we have
downstream policy-improvement results for each collection choice.

## The measured oracle

Every collection strategy in the study has a fine-tuned policy behind it, so
a participant's choice maps to a real downstream success rate. Numbers from
`$DATA_PATH/rollouts/*/episodes/metadata.yaml` (DiT runs):

| Task | Baseline | Interventions |
|---|---|---|
| set_table | `set_table_baseline_pretrained_dit` 32/55 = **58.2 %** | `+left` 46/60 = **76.7 %** (+18.5) · `+middle` 28/60 = 46.7 % (−11.5) · `+right` 26/60 = 43.3 % (−14.9) |
| prepare_tray | `prepare_tray_baseline` 37/60 = **61.7 %** | `+bottom` 54/60 = **90.0 %** (+28.3) · `+top` 47/60 = 78.3 % (+16.7) |
| cooking | `cooking_baseline_dit_200` 14/60 = **23.3 %** | `+lid` 16/60 = 26.7 % (+3.3) · `+carrot` 12/60 = 20.0 % (−3.3) · `+corn` 11/60 = 18.3 % (−5.0) |

**set_table is the strong task.** `+left` beats `+middle` at p = .0004 and
`+right` at p < .0001; both wrong choices land *below* baseline. A 33-point
spread between the best and worst decision, cleanly separated. This is the
task that carries the decision endpoint.

**prepare_tray is usable but thin.** Only two options, so chance is 50 %, and
`+bottom` vs `+top` is p ≈ .075. Fine as one of three, weak alone.

**cooking cannot carry a decision endpoint.** With `dit_200` as the baseline,
no intervention differs from the baseline or from any other: best vs worst is
`+lid` 26.7 % vs `+corn` 18.3 %, z = 1.10, p = .27, and every arm-vs-baseline
comparison is p > .5. Separating `+lid` from `+corn` at that effect size
(h ≈ 0.20) would need ~380 rollouts per arm. There is no measurably right
answer for a participant to find, so scoring a cooking collection decision
against downstream payoff would be scoring against noise.

**Run cooking as an interpretability-only task.** The core battery
(continuation prediction, failure-stage distribution, free text) needs only
the baseline rollouts plus an independent answer key — not the intervention
outcomes. And cooking is unusually good material for it: 46 failures out of
60 at a near-floor 23.3 % success means abundant failure structure to read,
and "how far does it get before breaking" is exactly what the graph shows.

*Possible rescue, conditional on annotation.* `carrot` / `corn` / `lid` look
like components of one multi-step task rather than per-episode conditions —
`args.instruction` is empty for every cooking run, so there is no condition
variable in the metadata. If the failure-stage annotation (blocker 1) shows
that `+lid` specifically reduces lid-stage failures, a *targeted* per-stage
effect could be significant even though the overall rate is not — the same
structure that makes set_table's numbers work. That is a reason to prioritise
the annotation work, not something to plan the study around.

Also relevant to task selection: the earlier `set_table_baseline_v2_sync`
family gives per-condition ground truth for the *baseline* policy —
left 0/20 (0 %), middle 6/20 (30 %), right 17/20 (85 %) — a clean monotone
structure that makes set_table's interpretability items easy to grade.

## What's already built

| Piece | Where | State |
|---|---|---|
| `order_effect_v1` variant | `configs/viz/study_variants/order_effect_v1.yaml` → `variants/order_effect/render.py` | Counterbalanced within-subject mockup. No consent step, no timer, intro still says "mockup". |
| Shared components | `user_study/` — `video_browser`, `graph_explorer`, `path_explorer`, `nasa_tlx`, `likert_survey`, `survey_steps`, `response_store` | Reusable; the order-effect variant already composes them rather than forking. |
| Randomisation | `apps/study/Home.py:47` | `random.choice(["A","B"])` per session. |
| Session configs | `configs/viz/study/` | set_table, lift, square, transport. **No prepare_tray or cooking.** |

## The design to run

**Modality between-subjects, task within-subjects, self-paced.**

Each participant is assigned to one arm (graph or videos) and does all three
tasks in that arm. Task order counterbalanced by Latin square.

- All three tasks contribute to the **interpretability** endpoints — roughly
  90 observations from 30 people under a mixed model with participant and
  task random intercepts.
- Only **set_table and prepare_tray** contribute to the **decision**
  endpoint, since cooking's interventions are statistically indistinguishable
  (see above). Two decisions per participant, ~60 decisions from 30 people.
- Task replication is the main defence against "you tuned this on one task",
  and it is worth more than another 20 participants on set_table alone.

**Drop the crossover.** Three tasks and a second-modality phase do not both
fit in a session (three tasks at 6–8 min plus intro and surveys is already
30–40 min). The crossover was carryover-contaminated anyway — once you have
seen the graph you cannot un-see the failure structure when you then watch
videos — so replication is the better use of the time budget.

**Decisions must stay between-subjects** regardless. A collection decision
can only be made once; making it again in the other modality is contaminated
by definition.

**Replace the fixed 600 s timer with self-pacing** (an "I'm ready to decide"
button, generous ~10 min cap, flag whoever hits it). This is the single most
important change to the existing flow. A diligent video-condition participant
*can* recover the answer by counting outcomes across 60 clips; the graph's
real advantage is that it surfaces the structure without that effort. The
fixed timer censors exactly that variable — everyone spends 600 s, so only
accuracy is observable, and if video participants grind it out the study
shows nothing. Self-pacing yields three correlated measures of one claim:

- accuracy — did they diagnose the failure correctly?
- time-to-decision — how long did it take?
- effort — NASA-TLX, already built.

"Same or better decisions in half the time at lower workload" is a stronger
and more robust result than a Likert delta, and it survives the case where
both arms ceiling on accuracy.

### Sample size

| Primary endpoint | Effect | n |
|---|---|---|
| Objective battery, continuous score | d ≈ 1.0 | ~17/arm |
| Decision regret (17.4 → 6.3 pts, pooled SD ≈ 14) | d ≈ 0.77 | ~27/arm |
| Diagnosis accuracy (.45 → .80) | h = 0.73 | ~30/arm |
| Likert self-report | d = 0.4 | ~100/arm |

Target **n = 50 total, 25 per arm**. Three tasks per participant and the
timing measure are what make that enough; a Likert-primary design at the same
n would be underpowered by a factor of four. Note the regret row assumes the
video arm splits 45/30/25 across set_table's options and the graph arm
80/10/10 — plausible, but the single biggest guess in this table.

## The question battery

The governing principle: **almost every question currently asked is
self-report about the experience, and we can now measure the thing itself.**
`likert_survey.py` (17 items) and `order_effect/questions.py` (5 items) are
dominated by "the graph gave me a clear overview", "I felt confident about
the strategies I selected", "the process felt intuitive". These are maximally
exposed to demand characteristics — the graph is visibly the treatment — and
reviewers discount them.

Structure as a **portable common core plus per-task specifics**, z-scored
within task and aggregated in the mixed model. Do not try to force one
battery onto all three tasks.

**Ask only what is answerable from the baseline rollouts.** Everything below
is a *counting* question about the current policy, never a forecast. A
participant can see that set_table's left condition is 0/20 and that right is
17/20; they cannot possibly infer how much a fine-tune will recover, whether
it spills over to other conditions, or that two of the three choices land
*below* baseline. Nobody could — those numbers weren't knowable before the
fine-tunes were run. The measured payoffs are **our scoring function, not a
question we put to participants**: they make an ordinal choice, and we grade
it against reality (see *Scoring decisions as regret*).

### Common core (all three tasks)

1. **Mid-episode continuation prediction.** Show the first half of ~10
   held-out episodes; ask whether each succeeds. Scored by balanced accuracy.
   This is the strongest interpretability measure because it operationalises
   understanding as predictive power, and unlike initial-frame prediction it
   works when initial conditions don't determine the outcome — which is the
   case for cooking. It is also exactly what the graph encodes: from this
   state, where do episodes go next.
2. **Failure-stage distribution.** "Of 100 failures, how many break during
   step 1 / 2 / 3?" Numeric estimates scored by error. Works on any
   multi-stage task and directly probes what the graph displays. Especially
   apt for cooking, which at 6.7–26.7 % success is a near-floor policy where
   the whole signal is how far it gets before breaking.
3. **Free text:** one sentence describing the policy's main failure mode.
   Protocol below.
4. **The decision:** ranked choice over the measured collection options,
   scored as regret against downstream payoff, plus **confidence**
   (→ calibration), **time-to-decision**, and **NASA-TLX**. set_table and
   prepare_tray only.

### Per-task specifics

- **set_table** — per-condition success-rate counts for the *baseline*
  policy: "of 20 attempts with the milk on the left, how many succeeded?"
  Truth 0 / 6 / 17. Scored by absolute error. High resolution, chance
  performance near zero, and it is the direct input to the collection
  decision. This is a set_table instrument; don't generalise it.
- **prepare_tray** — same shape with top/bottom, but only two levels.
- **cooking** — no decision item; core battery only. Weight the
  failure-stage distribution item most heavily here, since it is the measure
  best matched to a near-floor policy.

### The question worth hunting for

The strongest possible item is a **cross-episode structural discrimination**
— one where watching more videos does not substitute. For example: *"Do the
failures on left and the failures on middle share a common failure mode, or
are they different?"* With the graph you read it off the branch structure in
seconds; from videos you would have to hold forty clips in working memory.

Every other measure is vulnerable to "a diligent video participant could just
count harder". This one is not, which makes it the item that most directly
shows the graph is doing something rather than merely doing it faster. Worth
real effort to find one per task.

### What to cut

Block 1 of `likert_survey.py` (seven items on how the graph felt) and
`b2_q8_confidence`, `b2_q12_targeted`, `b2_q13_intuitive`. Keep two or three
as manipulation checks — enough to confirm engagement — and demote them from
endpoints.

`subtask1_primary_failure` / `subtask2_primary_failure` in the order-effect
bank are *factual* questions wearing a Likert costume. Rewrite as a forced
choice ("which subtask does the policy fail at most?") and they become
gradeable.

## Scoring decisions as regret

Express each participant's choice as **points of downstream policy success
left on the table**: `regret = payoff(best option) − payoff(chosen)`.

| Task | Option | Payoff | Regret (pts) | Uncertainty |
|---|---|---|---|---|
| set_table (base 58.2 %) | `+left` | 76.7 % | **0** | — |
| | `+middle` | 46.7 % | 30.0 | ± 8.4 |
| | `+right` | 43.3 % | 33.4 | ± 8.4 |
| prepare_tray (base 61.7 %) | `+bottom` | 90.0 % | **0** | — |
| | `+top` | 78.3 % | 11.7 | ± 6.6 |

- **Primary:** regret of the participant's top-ranked option. Unambiguous,
  and it assumes nothing about the value of mixtures (which we never
  measured).
- **Secondary:** Spearman correlation between their full ranking and the true
  value ordering — partial credit for near-misses.

### The chance line

Random choice has a computable expected regret: **21.1 points** on set_table,
**5.9** on prepare_tray. Plot it as a reference line, so the result reads:

> Graph arm 6.3 points forgone · videos arm 17.4 · random choice 21.1.

This answers the sharper question — *is the video condition even beating
random?* — and it comes free, with no third control arm to recruit.

### Caveats

**Scale heterogeneity.** set_table's maximum regret is 33.4 points,
prepare_tray's is 11.7, so a naive average lets set_table dominate roughly
3:1. Run the test on per-task-normalised regret (divide by the task maximum);
report raw points per task for the headline.

**prepare_tray's regret is not confidently nonzero.** 11.7 ± 6.6 (p ≈ .075)
means we cannot firmly claim `+top` was actually the worse choice. set_table
carries the decision endpoint nearly alone; prepare_tray adds a noisy second
observation.

**Regret is not extra statistical power here.** The two wrong set_table
options are nearly equally bad (30.0 vs 33.4), so regret is close to a
relabelling of binary accuracy. What it buys is *units a reader understands* —
points of downstream policy success rather than proportion-correct — plus the
chance-line comparison. Present it as an interpretability win, not a
power win.

**Diagnosis and decision are coupled.** On set_table, once you know left is at
0/20 the choice follows almost deterministically. Don't present the diagnosis
battery and the decision endpoint as two independent confirmations — the
decision is largely a readout of the diagnosis. This is arguably a feature,
but a reviewer will notice. It also means the interpretability battery
carries more weight, since it is the part with independent variance across
all three tasks.

## Free-text evaluation protocol

**Score three dimensions, 0–2 each, summed to 0–6:**

- **Localisation** — does it name the stage where failures actually
  concentrate?
- **Trigger** — does it name what distinguishes failing from succeeding
  episodes?
- **Specificity** — is the claim falsifiable? "It fails when reaching for the
  milk on the left" scores 2; "it struggles with grasping" scores 0.

Specificity matters more than it looks: it is gradeable *without* an answer
key, and it captures the difference we most expect — graph users producing
specific structural claims, video users producing vague impressions. It
yields a result even where the answer key is weak.

**Write the rubric before looking at responses**, or it will end up rewarding
whatever graph users happened to say.

**Blinding fails unless vocabulary is handled.** Graph users will echo the VLM
node labels ("it enters the reach-shelf mode"); video users won't. Any coder
infers the condition instantly. Either have an LLM paraphrase every response
into neutral phrasing before coding, or restrict the rubric to factual
content and accept partial blinding — and state which.

**Scale with validated LLM coding.** At 3 tasks × 40 participants that's ~120
responses. Have an LLM score all of them blind against the rubric, have two
humans independently score a random 25 %, and report Cohen's κ between LLM
and human. If κ > 0.7, use the LLM scores throughout and report the
validation.

**Optional, high-ceiling:** give a separate group of naive raters *only* the
free-text descriptions and ask them to predict held-out episode outcomes. If
descriptions written by graph users let a third party predict better, that
shows the graph produces *transmissible* understanding, not just understanding
people report having.

## Blockers and fixes

Ordered by what gates what.

**1 — There is no answer key.** Every DiT run carries only `success` and
`length`; `note` is empty throughout; and length does not separate outcomes
(cooking 798 vs 748 frames, prepare_tray 512 vs 518, set_table 408 vs 495).
Both portable core measures need per-episode failure-stage annotation that
does not currently exist. Roughly: define 3–5 stage boundaries per task from
the task script, sample the failures, label them — VLM-assisted with human
validation on a subsample, using the existing annotation infrastructure.
**This gates the whole study build.**

**2 — Do not derive the answer key from the graph.** If failure stage is
defined by which node an episode terminates in, participants are being graded
against the tool under test. The key must come from independent annotation.
(Mild circularity in *item selection* — picking prefix points where the graph
shows a real branch — is far less damaging than in the *answer key*, but
pre-register the selection rule.)

**3 — Condition labels are missing from the DiT runs.** `args.instruction` is
empty across every `set_table_*_dit`, every cooking run, and
`prepare_tray_plus{top,bottom}`. Only `prepare_tray_baseline_dp` and the
v2_sync family carry them. For set_table the convention is recoverable —
v2_sync is contiguous blocks of 20 (`right×20, middle×20, left×20`) — but
**do not assume it transfers**: `set_table_baseline_pretrained_dit` has 55
episodes, not 60, so five were dropped and the block boundaries shifted.
Verify per run.

**4 — Cut the option set to what was measured.** `set_table_baseline.yaml`
offers 9 strategies, 6 of them `s_perm_*` permutations with no measured
outcome — unscoreable. Worse, three of the nine contain milk-left, so "milk-
left appears in the top 3" is satisfied by *random* ranking 76 % of the time.
That measure cannot detect anything. Reduce to the three measured options:
chance becomes 1/3 on first choice and 1/6 on the full ranking.

**5 — The blinded 3-clip ranking leaks.** `variants/order_effect/render.py:70–86`
shows a *failure* clip for left and *successes* for middle and right, so a
participant who learned nothing can still pick "the one that failed". Use
initial-condition stills or human demo clips instead.

**6 — Randomisation and data integrity.** `random.choice` per session gives
meaningfully unbalanced arms at n = 40, and a hard browser reload appears to
re-roll assignment and lose partial progress. Use permuted blocks of 4
persisted alongside the response store, and persist partial state keyed by
participant ID.

**7 — Attention checks.** At 25–35 min online, expect 10–20 % straight-lining.
One instructed-response item plus a minimum-time criterion buys more effective
power than ten extra participants.

**8 — Group B is not just "plus graph".** It's graph + path explorer +
failure-rate bar chart + per-node clip slicing (the graph display, sections
B–D). A reviewer will say the bar chart did the work. Accept this and frame it
honestly as system-vs-baseline rather than spending 3× the sample on an
ablation arm — but say so explicitly.

**9 — Missing session configs.** `configs/viz/study/` has no prepare_tray or
cooking entries. Each needs a session yaml and a graph run; prepare_tray also
needs a strategy option set matching its two measured interventions. Cooking
needs no strategy set — it runs interpretability items only.

**10 — Housekeeping.** The order-effect variant has no consent step, and
`_render_intro` still says "Order-Effect Study Mockup" / "This is a mockup of
a study."

## Analysis plan

- **Primary:** objective core battery, graph vs videos, mixed model with
  participant and task random intercepts, all three tasks.
- **Secondary:** decision regret in points of downstream policy success,
  per-task-normalised for the test and raw for reporting, against the
  random-choice line (set_table and prepare_tray only); time-to-decision;
  NASA-TLX.
- **Tertiary:** calibration (confidence × accuracy); free-text rubric score;
  the Likert manipulation checks.

Report regret in real units as the interpretable headline, but hang the
significance test on the diagnosis battery — it has independent variance
across all three tasks, whereas regret rests almost entirely on set_table.
Keep the two roles distinct.

## Open questions

- **Target population — naive online participants or robotics
  practitioners?** If practitioners, recruitment caps around n = 20–25 and the
  objective battery stops being merely better and becomes the only thing that
  will reach significance.
- **Is one decision per task enough, given only two decision tasks?** With
  cooking out, the decision endpoint rests on 2 observations per participant.
  Worth considering a second decision round on set_table after a reveal (show
  the participant what their choice actually did, using the `plus*` rollouts,
  then re-decide) — it doubles decision observations and measures belief
  updating, at the cost of contaminating the second round.
