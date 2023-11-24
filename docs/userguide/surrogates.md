# Surrogates

Surrogate models are used to model and estimate the unknown objective function of the DoE campaign. BayBE offers a diverse array of surrogate models, while also allowing for the utilization of custom models. All surrogate models are based upon the general [`Surrogate`](baybe.surrogates.base.Surrogate) class. Some models even support transfer learning, as indicated by the `supports_transfer_learning` attribute.

## Available models

BayBE provides a comprehensive selection of surrogate models, empowering you to choose the most suitable option for your specific needs. The following surrogate models are available within BayBE:

* [`GaussianProcessSurrogate`](baybe.surrogates.gaussian_process.GaussianProcessSurrogate)
* `BayesianLinearSurrogate`
* [`MeanPredictionSurrogate`](baybe.surrogates.naive.MeanPredictionSurrogate)
* `NGBoostSurrogate`
* `RandomForestSurrogate`


## Using custom models

BayBE goes one step further by allowing you to incorporate custom models based on the ONNX architecture. Note however that these cannot be retrained.  For a detailed explanation on using custom models, refer to the comprehensive examples provided in the corresponding [example folder](./../../examples/Custom_Surrogates/Custom_Surrogates).