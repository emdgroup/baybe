# Active Learning
When labeling unmeasured data points, e.g. for a data acquisition campaign to gather 
data for a machine learning model, it can be beneficial to follow a guided approach 
rather than randomly measuring data. If this is done via iteratively measuring points 
according to a criterion proportional to the current model's uncertainty, the 
method is called **active learning**.

Active learning can be seen as a special case of Bayesian optimization. If we have the 
above-mentioned criterion and set up a Bayesian optimization campaign to recommend 
points with the highest uncertainty, we achieve active learning via Bayesian 
optimization. In practice, we can leverage the fact that we have a posterior 
distribution, and compute an acquisition function that represents the overall 
uncertainty about any given point in the searchspace.

Below you find which acquisition functions in BayBE are suitable for this endeavor 
including a few guidelines. Furthermore, BayBE offers you the ability to choose a 
custom surrogate model. This means you can use the model you intend to measure points 
for already during the data acquisition phase (more on this soon).

## Single-Point Uncertainty
In BayBE, there are two acquisition functions that can be chosen to search for the 
points with the highest predicted model uncertainty:
- [`PosteriorStandardDeviation`](baybe.acquisition.acqfs.PosteriorStandardDeviation)
- [`UpperConfidenceBound`](baybe.acquisition.acqfs.UpperConfidenceBound) / 
  [`qUpperConfidenceBound`](baybe.acquisition.acqfs.qUpperConfidenceBound) with high 
  `beta`: A high `beta` will effectively cause that the mean part of the acquisition 
  function becomes irrelevant, and it only searches for points with the highest 
  posterior variability. We recommend values `beta > 10.0`.

## Integrated Uncertainty
BayBE also offers the 
[`qNegIntegratedPosteriorVariance`](baybe.acquisition.acqfs.qNegIntegratedPosteriorVariance) 
(short [`qNIPV`](baybe.acquisition.acqfs.qNegIntegratedPosteriorVariance)). It integrates 
the posterior variance on the entire searchspace. If your campaign selects points based 
on this acquisition function, it thus chooses the ones which contribute most to the 
current global model uncertainty and recommends them for measurement. This approach is 
often superior to using a single point uncertainty estimate as acquisition function.

Due to the computational complexity, it can be prohibitive to integrate over the entire 
searchspace. For this reason we offer the ability to sub-sample parts of it, configured 
in `qNIPV`:

```python
from baybe.acquisition import qNIPV
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

# Will integrate over the entire searchspace
qNIPV()

# Will integrate over 50% of the searchspace, randomly sampled
qNIPV(sampling_fraction=0.5)

# Will integrate over 100 points, chosen by farthest point sampling
# Both lines are equivalent
qNIPV(sampling_n_points=100, sampling_method="FPS")
qNIPV(sampling_n_points=100, sampling_method=DiscreteSamplingMethod.FPS)
```

**Sampling of the continuous part of the searchspace will always be random**, while 
sampling of the discrete part can be controlled by providing a corresponding 
[`DiscreteSamplingMethod`](baybe.utils.sampling_algorithms.DiscreteSamplingMethod) for 
`sampling_method`.

```{admonition} Purely Continuous SearchSpaces
:class: important
Please be aware that in case of a purely continuous searchspace, the number of points 
to sample for integration must be specified via `sampling_n_points`.
```