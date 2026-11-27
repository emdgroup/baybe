# Optimizing complete multi-step configurations

Use a complete-configuration campaign when the same parameter assignment should be
carried through every step of a multi-step DAG. The existing
{meth}`~baybe.multistep.MultiStepCampaign.recommend` API remains the local node-action
workflow; it does not change into a full-configuration optimizer.

## Initial configurations

Choose initial configurations explicitly with
{meth}`~baybe.multistep.MultiStepCampaign.sample_configuration`. It independently
samples each owned parameter independently within its bounds or allowed values, then
reserves the result under one experiment identifier. Execute the reserved assignment
with {func}`~baybe.multistep.evaluate_configuration` and the evaluator already used by
your application:

~~~python
from baybe.multistep import evaluate_configuration

configuration = campaign.sample_configuration(sampling_seed=13)
experiment_id = configuration.loc[0, "experiment_id"]
evaluate_configuration(
    campaign,
    experiment_id,
    lambda action: benchmark.evaluate_multistep_actions(action),
)
~~~

The callback receives one local action at a time. It must return one measurement row
for that action. Each valid row is stored immediately; the next action reuses the
reserved parameters and only real parent/ancestor measurements. A failed step stops
that attempt while retaining its actual outputs. Technical exceptions leave the
current action pending and propagate to the caller, so a later explicit call can
resume it without repeating completed steps.

## Bayesian recommendations

Call {meth}`~baybe.multistep.MultiStepCampaign.recommend_configuration` only after the
campaign has at least one complete, successful real process measurement. Incomplete
or failed attempts are not EI references. They are never filled with a synthetic
terminal value, and failed initial attempts are not silently replaced.

~~~python
configuration = campaign.recommend_configuration(
    acquisition_function="qLogEI",
    sampling_seed=27,
)
experiment_id = configuration.loc[0, "experiment_id"]
evaluate_configuration(
    campaign,
    experiment_id,
    lambda action: benchmark.evaluate_multistep_actions(action),
)
~~~

The method acquisition (`qLogEI` by default, or `qEI`) is separate from the
campaign's acquisition and recommender settings for local node actions. Its EI
reference is the maximum objective-transformed network posterior mean over complete
successful process measurements, following BayBE's existing `best_f` semantics;
it is not the maximum observed utility. The optimizer selects one complete numeric
configuration and reserves it before execution.

Inspect actual results in the campaign data store, for example with
`campaign.data.complete_measurement_frame(campaign.dag, execution_mode="process")`.
This view contains complete successful measurements only; use the measurement store
for real partial outputs and failure labels.

## Scope and limitations

- The full-configuration API supports numerical continuous and numerical discrete
  parameters. Categorical parameters remain available for fixed registration and
  random sampling, but not for numerical EI optimization.
- Configure process execution, `allow_repeats=True`, and `OBSERVED` or `EXCLUDE`
  failure training modes. Only one incomplete process attempt may be active.
- The callback workflow is synchronous and executes one local action at a time. It
  does not refit or re-recommend between nodes.
- Ordinary qEI/qLogEI optimizes modeled terminal utility only; it does not multiply
  the score by failure probability or process cost. It does not claim a global
  optimum.
- qLogEI uses BoTorch's smoothed logarithmic improvement, not the exact logarithm of
  qEI. Posterior means and samples are evaluated in the model's existing target
  scale.
