# Strategies

Strategies play a crucial role in orchestrating the usage of recommenders within a campaign. A strategy operates on a sequence of recommenders and determines when to switch between them. All strategies are built upon the[`Strategy`](baybe.strategies.base.Strategy) class.

BayBE offers three distinct kinds of strategies.

## The [`SequentialStrategy`](baybe.strategies.composite.SequentialStrategy) 

The `SequentialStrategy` introduces a simple yet versatile approach by utilizing a predefined list of recommenders. By specifying the desired behavior using the `mode` attribute, it is possible to flexibly determine the strategy's response when it exhausts the available recommenders. The possible choices are to either raise an error, re-us the last recommender or re-start at the beginning of the sequence.

## The [`StreamingSequentialStrategy`](baybe.strategies.composite.StreamingSequentialStrategy)

Similar to the `SequentialStrategy`, the `StreamingSequentialStrategy` enables the utilization of *arbitrary* iterables to select recommender. Note that this strategy is however not serializable.

## The [`TwoPhaseStrategy`](baybe.strategies.composite.SequentialStrategy)

The `TwoPhaseStrategy` employs two distinct recommenders and switches between them at a certain specified point, controlled by the `switch_after` attribute.

## Additional options for discrete search spaces

For discrete search spaces, BayBE provides additional control over strategies. You can explicitly define whether a strategy is allowed to recommend previously used recommendations and whether it can output recommendations that have already been measured. 