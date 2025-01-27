# Surrogates

Surrogate models are used to model and estimate the unknown objective function of the
DoE campaign. BayBE offers a diverse array of surrogate models, while also allowing for
the utilization of custom models. All surrogate models are based upon the general
[`Surrogate`](baybe.surrogates.base.Surrogate) class. Some models even support transfer
learning, as indicated by the `supports_transfer_learning` attribute.

## Available Models

BayBE provides a comprehensive selection of surrogate models, empowering you to choose
the most suitable option for your specific needs. The following surrogate models are
available within BayBE:

* [`GaussianProcessSurrogate`](baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate)
* [`BayesianLinearSurrogate`](baybe.surrogates.linear.BayesianLinearSurrogate)
* [`MeanPredictionSurrogate`](baybe.surrogates.naive.MeanPredictionSurrogate)
* [`NGBoostSurrogate`](baybe.surrogates.ngboost.NGBoostSurrogate)
* [`RandomForestSurrogate`](baybe.surrogates.random_forest.RandomForestSurrogate)

## Extracting the Model for Advanced Study

In principle, the surrogate model does not need to be a persistent object during
Bayesian optimization since each iteration performs a new fit anyway. However, for
advanced study, such as investigating the posterior predictions, acquisition functions
or feature importance, it can be useful to diretly extract the current surrogate model.

For this, BayBE provides the ``get_surrogate`` method, which is available for the
[``Campaign``](baybe.campaign.Campaign.get_surrogate) or for 
[recommenders](baybe.recommenders.pure.bayesian.base.BayesianRecommender.get_surrogate).
Below an example of how to utilize this in conjunction with the popular SHAP package:

~~~python
# Assuming we already have a campaign created and measurements added
data = campaign.measurements[[p.name for p in campaign.parameters]]
model = lambda x: campaign.get_surrogate().posterior(x).mean

# Apply SHAP
explainer = shap.Explainer(model, data)
shap_values = explainer(data)
shap.plots.bar(shap_values)
~~~

```{admonition} Current Scalarization Limitations
:class: note
Currently, ``get_surrogate`` always returns the surrogate model with respect to the
transformed target(s) / objective. This means that if you are using a
``SingleTargetObjective`` with a transformed target or a ``DesirabilityObjective``, the
model's output will correspond to the transformed quantities and not the original
untransformed target(s). If you are using the model for subsequent analysis this should
be kept in mind.
```

## Using Custom Models

BayBE goes one step further by allowing you to incorporate custom models based on the
ONNX architecture. Note however that these cannot be retrained. For a detailed
explanation on using custom models, refer to the comprehensive examples provided in the
corresponding [example folder](./../../examples/Custom_Surrogates/Custom_Surrogates).