# Active Learning
When deciding which experiments to perform next, e.g. for a data acquisition campaign
to gather data for a machine learning model, it can be beneficial to follow a guided
approach rather than selecting experiments randomly. If this is done via iteratively
measuring points according to a criterion reflecting the current model's uncertainty,
the method is called **active learning**.

Active learning can be seen as a special case of Bayesian optimization: If we have the
above-mentioned criterion and set up a Bayesian optimization campaign to recommend
points with the highest uncertainty, we achieve active learning via Bayesian
optimization. In practice, this is procedure is implemented by setting up a
probabilistic model of our measurement process that allows us to quantify uncertainty
in the form of a posterior distribution, from which we can then construct an
uncertainty-based acquisition function to guide the exploration process.

Below you find which acquisition functions in BayBE are suitable for this endeavor, 
including a few guidelines.

## Local Uncertainty Reduction
In BayBE, there are two types of acquisition function that can be chosen to search for
the points with the highest predicted model uncertainty:
- [`PosteriorStandardDeviation`](baybe.acquisition.acqfs.PosteriorStandardDeviation) (`PSTD`)
- [`UpperConfidenceBound`](baybe.acquisition.acqfs.UpperConfidenceBound) (`UCB`) / 
  [`qUpperConfidenceBound`](baybe.acquisition.acqfs.qUpperConfidenceBound) (`qUCB`)
  with high `beta`:  
  Increasing values of `beta` effectively eliminate the effect of the posterior mean on
  the acquisition value, yielding a selection of points driven primarily by the
  posterior variance. However, we generally recommend to use this acquisition function
  only if a small exploratory component is desired – otherwise, the
  [`PosteriorStandardDeviation`](baybe.acquisition.acqfs.PosteriorStandardDeviation) 
  acquisition function is what you are looking for.

## Global Uncertainty Reduction
BayBE also offers the 
[`qNegIntegratedPosteriorVariance`](baybe.acquisition.acqfs.qNegIntegratedPosteriorVariance) 
(`qNIPV`), which integrates 
the posterior variance over the entire search space.
Choosing candidates based on this acquisition function is tantamount to selecting the
set of points resulting in the largest reduction of global uncertainty when added to
the already existing experimental design.

Because of its ability to quantify uncertainty on a global scale, this approach is often
superior to using a point-based uncertainty criterion as acquisition function. 
However, due to its computational complexity, it can be prohibitive to integrate over
the entire search space. For this reason, we offer the option to sub-sample parts of it,
configurable via the constructor:

```python
from baybe.acquisition import qNIPV
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

# Will integrate over the entire search space
qNIPV()

# Will integrate over 50% of the search space, randomly sampled
qNIPV(sampling_fraction=0.5)

# Will integrate over 250 points, chosen by farthest point sampling
# Both lines are equivalent
qNIPV(sampling_n_points=250, sampling_method="FPS")
qNIPV(sampling_n_points=250, sampling_method=DiscreteSamplingMethod.FPS)
```

```{admonition} Sub-Sampling Method
:class: note
Sampling of the continuous part of the search space will always be random, while 
sampling of the discrete part can be controlled by providing a corresponding 
[`DiscreteSamplingMethod`](baybe.utils.sampling_algorithms.DiscreteSamplingMethod) for 
`sampling_method`.
```

```{admonition} Purely Continuous Search Spaces
:class: important
Please be aware that in case of a purely continuous search space, the number of points 
to sample for integration must be specified via `sampling_n_points` (since providing
a fraction becomes meaningless).
```