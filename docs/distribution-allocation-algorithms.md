# Distribution Allocation Algorithms

This report describes the integer allocation, or apportionment, problem behind dataset class target calculation.

The problem is that ideal class quotas are often fractional, while generated or sampled datasets need whole-number counts that usually sum exactly to a fixed `target_size`.

In the current implementation, `AbstractDistributionCalculator.percentage_targets` uses the largest remainder method for percentage-based targets. It computes exact fractional quotas, floors each quota, then gives the remaining units to the classes with the largest decimal remainders. Ties are shuffled through the configured `random.Random` instance so classes with equal fractional remainders are not always favored by label order.

`TargetDatasetDistributionCalculator` returns those calculated targets unchanged. `SourceDatasetDistributionCalculator` applies source availability constraints afterward by clamping targets to `dataset_class_count` and redistributing leftover units to classes that still have capacity.

## 1. Largest Remainder Method

This is what the percentage-based code currently does before capacity constraints are applied.

Algorithm:

1. Compute the exact fractional quota for each class.
2. Floor each exact quota to get an initial integer target.
3. Compute the leftover count: `target_size - sum(initial_targets.values())`.
4. Assign leftover units to classes with the largest fractional remainders, shuffling tied remainder groups.

Example:

```text
target_size = 10
percentages = {A: 33.3, B: 33.3, C: 33.4}

exact quotas = {A: 3.33, B: 3.33, C: 3.34}
floors       = {A: 3,    B: 3,    C: 3}
remainder    = 1
final        = {A: 3,    B: 3,    C: 4}
```

Pros:

- Simple and intuitive.
- Guarantees the final total equals `target_size` before source-capacity clamping.
- Each class usually gets either `floor(exact)` or `ceil(exact)`.
- Very easy to explain and debug.

Cons:

- Can behave oddly when `target_size` changes slightly.
- Ties need explicit handling, like the current shuffle.
- It optimizes local rounding error, not necessarily global fairness across repeated runs.
- It is known in apportionment theory to suffer from paradoxes, such as the Alabama paradox: increasing the total number of seats or items can sometimes make a class lose one.

This is usually a good choice for dataset class balancing because it is transparent and predictable.

## 2. Standard Rounding Then Adjustment

This method rounds each exact target normally, then fixes the sum if it is too high or too low.

Algorithm:

1. Compute the exact fractional quota for each class.
2. Round each quota to the nearest integer.
3. Compare the rounded total with `target_size`.
4. Add to classes with the biggest positive rounding loss, or subtract from classes with the weakest rounding justification.

If the rounded total is too small, add units to the classes that lost the most during rounding. If the rounded total is too large, subtract units from classes that were rounded up with the smallest justification.

Pros:

- Matches normal human intuition about rounding.
- Often gives lower total absolute rounding error than flooring-first methods.
- Easy to implement.

Cons:

- The rounded values may not sum to `target_size`.
- Requires a second correction pass.
- Correction can feel arbitrary unless carefully designed.
- A class can be pushed away from its natural rounded value during adjustment.

This is useful when conventional rounding behavior matters more than strict floor-then-fill logic.

## 3. Divisor Methods

Divisor methods are used in political seat apportionment. Instead of flooring once, they search for a divisor that makes rounded quotas sum to the target.

General idea:

1. Choose an initial divisor.
2. Divide each class weight by that divisor.
3. Round each adjusted quota according to the selected method.
4. Adjust the divisor until the rounded allocations sum to `target_size`.

The target condition is:

```text
sum(round(weight_i / divisor)) = target_size
```

Variants differ by how they round:

- Jefferson / D'Hondt: uses floor.
- Adams: uses ceiling.
- Webster / Sainte-Lague: uses standard rounding.
- Huntington-Hill: uses geometric mean thresholds.

Pros:

- Well-studied and principled.
- Often better for proportional allocation at scale.
- Some variants avoid specific paradoxes that affect largest remainder.
- Good when allocations represent seats, capacity, budget, or other formal quotas.

Cons:

- Less obvious to readers.
- More complex to implement correctly.
- Some variants systematically favor large classes, while others favor small classes.
- May assign a class below its floor or above its ceiling depending on the variant and constraints.

For dataset generation, divisor methods may be overkill unless class proportions matter very strongly.

## 4. Stochastic Rounding

Stochastic rounding gives each class either `floor(exact)` or `ceil(exact)` based on the fractional part as a probability.

Example:

```text
exact quota for A = 3.7

A gets 4 with probability 0.7
A gets 3 with probability 0.3
```

Algorithm:

1. Compute the exact fractional quota for each class.
2. Assign each class its floor.
3. Treat each fractional remainder as a probability.
4. Randomly round each class up or down according to that probability.

Pros:

- Fair in expectation over many runs.
- Avoids deterministic bias.
- Natural if the whole generation process is already random.
- Good for repeated sampling pipelines.

Cons:

- One individual run may not match the desired proportions very closely.
- Does not automatically guarantee the final sum equals `target_size`.
- Needs extra coordination if exact total size is mandatory.
- Harder to test unless seeded.

A controlled version can guarantee the total by sampling exactly `remainder` classes weighted by fractional parts.

## 5. Weighted Random Allocation

Weighted random allocation floors every exact target first, then assigns remaining slots randomly with probabilities proportional to the fractional remainders.

Algorithm:

1. Compute the exact fractional quota for each class.
2. Assign each class its floor.
3. Compute the leftover count.
4. Choose `remainder` classes using weights proportional to their fractional remainders.

Pros:

- Reduces deterministic bias.
- Classes with larger fractional parts are more likely to receive extras.
- Fairer across many repeated dataset generations.

Cons:

- A class with the largest fraction is not guaranteed to receive the extra item.
- More variable results.
- Slightly harder to reason about than largest remainder.
- Requires care to avoid selecting the same class twice if each class can receive only one extra.

This is useful when generating many datasets and optimizing for long-run fairness rather than one-run determinism.

## 6. Minimum / Maximum Constrained Allocation

Some classes have availability constraints, like the current `dataset_class_count` concept. The algorithm must respect bounds:

```text
min_i <= allocation_i <= max_i
```

For example, a source dataset might not contain enough examples of class `A`, so even if the ideal target is 10, the maximum possible target is 6.

The current source calculator follows this pattern:

1. Calculate unconstrained targets through balanced, random, or percentage allocation.
2. Clamp each class target to its source availability in `dataset_class_count`.
3. Compute how many target units were lost during clamping.
4. Redistribute those units across classes with remaining capacity, weighted by available capacity.

Pros:

- Handles real-world dataset limits.
- Prevents impossible targets.
- More robust when source data is imbalanced.

Cons:

- More complex.
- If one class cannot absorb its quota, leftover count must be redistributed.
- Can drift away from desired percentages.
- Needs clear behavior when the requested `target_size` is impossible.

This matters for `SourceDatasetDistributionCalculator`, where source availability is part of the problem. In the current implementation, a `ValueError` is raised when `target_size` exceeds total available source dataset capacity.

## 7. Optimization-Based Allocation

The allocation can also be formulated as an optimization problem.

Minimize total absolute error:

$$
\min \sum_i |x_i - q_i|
$$

Subject to:

$$
\sum_i x_i = N
$$

and:

$$
x_i \in \mathbb{Z}
$$

where $q_i$ is the exact fractional target and $x_i$ is the integer allocation.

You can also minimize squared error:

$$
\min \sum_i (x_i - q_i)^2
$$

Pros:

- Very flexible.
- Can include min/max constraints, penalties, priorities, protected classes, and other business rules.
- Gives a clear mathematical objective.
- Good for complex allocation policies.

Cons:

- More machinery.
- Less readable than a direct algorithm.
- Might require an optimization library for advanced constraints.
- Overkill for simple class balancing.

For the current use case, this is probably unnecessary unless the balancing rules grow more complex.

## Practical Recommendation

For this codebase, the most relevant choices are:

| Algorithm | Best when |
| --- | --- |
| Largest remainder | You want simple, deterministic, explainable percentage-to-count conversion. |
| Largest remainder with shuffled ties | You want explainability plus no label-order bias, which is what the current percentage allocation does. |
| Weighted random remainder | You generate many datasets and want long-run probabilistic fairness. |
| Constrained largest remainder | You need to respect source dataset availability, which is what the source calculator does after unconstrained allocation. |
| Optimization-based allocation | You have multiple constraints and business rules. |

The current algorithm is a good default for target dataset generation. It is transparent, easy to test with a seeded random generator, and keeps the total equal to `target_size`.

For source dataset allocation, the current design already adds the important next layer: `dataset_class_count` capacity constraints. The tradeoff is that clamping and weighted redistribution can move the final allocation away from the original percentage intent when source data is imbalanced. That behavior is appropriate as long as the priority is to satisfy `target_size` while never requesting more examples from a class than the source dataset contains.