# RGPE — design notes

A short summary of the discussion behind the RGPE transfer-learning surrogate.

## What RGPE is

RGPE (Rank-weighted GP Ensemble) is a transfer-learning method. The idea in one
breath:

1. Split the data by task (the values of the `TaskParameter`).
2. Train one ordinary GP per task, each on its own data with the task column removed.
3. Give each GP a **weight** that says how much we trust it for the target task.
4. The final surrogate is a **weighted blend** of those GPs.

The weights come from a simple "ranking" test: a model is trusted more if the way it
ranks the target points agrees with how the real measurements rank them. The target
task's own model is scored fairly using leave-one-out (predict each point from the
others).

## Two decisions we had to make

We found it helped to separate two independent questions.

### Axis 1 — How do we hold and train the per-task GPs?

- **A loop (chosen).** Train the GPs one after another and keep them in a list.
  Simple, and it reuses BayBE's existing GP as-is (kernels, priors, scaling).
- **Stacked / batched GP.** Train all tasks at once using a batch dimension. Tempting
  for speed, but: it needs the same number of points per task (rarely true), the batch
  slot is already used by other machinery, tasks can bleed together, and we'd still
  have to write the blending code anyway. More risk, little gain.
- **`ModelListGP`.** A tidy container for several GPs, but it treats each GP as a
  separate *output*, which is the wrong shape for a weighted blend — we'd unpack it and
  re-blend by hand. No real benefit over the list.

**Verdict:** the loop. (Batching *is* used in one small spot — the target's
leave-one-out step — because there every fold has the same size.)

### Axis 2 — Where does the ensemble logic live?

- **A new surrogate (chosen):** `RGPETransferSurrogate`. All the RGPE logic lives in
  one focused place, separate from the plain GP.
- **Inside the existing GP factory.** Tempting because it already knows the search
  space, but it would bloat the single-model GP with a list of models, weights and a
  branching posterior — two personalities in one class, and messier to save/load.

**Verdict:** a new surrogate, with only a **thin dispatch** left in the GP factory that
hands off when the task parameter asks for RGPE.

## Other considerations

- **Choosing the method (`tl_mode`).** The plan is a small setting on the
  `TaskParameter` (`INDEX_KERNEL` vs `RGPE`) so the search space tells the BO loop which
  transfer-learning method to use. A thin dispatch in the GP factory reads it and hands
  off to the RGPE surrogate.
- **Exposing the model to acquisition (`to_botorch`).** The optimizer needs a real
  BoTorch model. We hold the per-task GPs in a `ModelListGP` (independent outputs) and
  turn them into the RGPE blend with a `ScalarizedPosteriorTransform`: because the
  outputs are independent, scalarizing with the weights gives exactly `mean = Σ wᵢ·μᵢ`
  and `cov = Σ wᵢ²·Σᵢ`. This avoids a hand-written blend model *and*, by inheriting from
  `ModelListGP`, keeps `fantasize`/`condition_on_observations` working — which the
  look-ahead acquisition functions (qKG, qNIPV) need. The one fiddly bit left is
  re-wrapping the fantasized model so the scalarization survives. (A simpler
  alternative — extending `AdapterModel` to apply a `posterior_transform` — only
  unlocks the analytic acquisition functions, not the look-ahead ones.)
- **A task-free search space.** To fit each per-task GP we need a search space without
  the task parameter that still supports `transform`. The existing `_drop_parameters`
  helper returns a reduced space that *blocks* `transform`, so we widened its allow-list
  to expose the few attributes fitting needs.

## Still open

- How exactly to build the task-free search space (extend `_drop_parameters` vs rebuild).
- Whether the dispatch stays in the GP factory or moves to the recommender level.
- Weight dilution when there are very many source tasks (the tutorial skips this).
