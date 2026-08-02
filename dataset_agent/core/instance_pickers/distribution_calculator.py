import random
from abc import ABC, abstractmethod
from collections.abc import Mapping


class AbstractDistributionCalculator(ABC):
    """Base class for calculating target counts per dataset class.

    Parameters
    ----------
    class_labels : set[str]
        Labels that should receive a target count.
    target_size : int
        Total number of instances to distribute across ``class_labels``.
    rng : random.Random
        Random number generator used to break ties and assign random targets.
    """

    def __init__(self, class_labels: set[str], target_size: int, rng: random.Random):
        self.class_labels = sorted(class_labels)
        self.target_size = target_size
        self.rng = rng

    def balanced_targets(self) -> dict[str, int]:
        """Calculate a balanced target count for each class.

        Returns
        -------
        dict[str, int]
            Target count for each class after capacity constraints are applied.

        Raises
        ------
        ValueError
            If capacity constraints cannot satisfy ``target_size``.
        """
        targets = self._balanced_unconstrained_targets()
        return self._apply_capacity_constraints(targets)

    def random_targets(self) -> dict[str, int]:
        """Calculate a randomized target count for each class.

        Returns
        -------
        dict[str, int]
            Target count for each class after capacity constraints are applied.

        Raises
        ------
        ValueError
            If ``target_size`` is smaller than the number of classes, or if
            capacity constraints cannot satisfy ``target_size``.
        """
        targets = self._random_unconstrained_targets()
        return self._apply_capacity_constraints(targets)
    
    def percentage_targets(self, percentages: Mapping[str, float]) -> dict[str, int]:
        """Calculate target counts from percentage allocations.

        Parameters
        ----------
        percentages : Mapping[str, float]
            Percentage allocation for each class label.

        Returns
        -------
        dict[str, int]
            Target count for each class after capacity constraints are applied.

        Raises
        ------
        KeyError
            If ``percentages`` does not contain an entry for every class label.
        ValueError
            If capacity constraints cannot satisfy ``target_size``.
        """
        targets = self._percentage_unconstrained_targets(percentages)
        return self._apply_capacity_constraints(targets)

    @abstractmethod
    def _apply_capacity_constraints(self, calculated_targets: dict[str, int]) -> dict[str, int]:
        """Apply implementation-specific capacity constraints.

        Parameters
        ----------
        calculated_targets : dict[str, int]
            Unconstrained target count for each class.

        Returns
        -------
        dict[str, int]
            Target count for each class after capacity constraints are applied.

        Raises
        ------
        ValueError
            If the implementation cannot satisfy ``target_size`` under its
            capacity constraints.
        """
        ...

    def _balanced_unconstrained_targets(self) -> dict[str, int]:
        """Calculate balanced targets without capacity constraints.

        Returns
        -------
        dict[str, int]
            Balanced target count for each class.
        """
        min_occurrences = self.target_size // len(self.class_labels)
        remainder = self.target_size % len(self.class_labels)
        remainder_sample = self.rng.sample(self.class_labels, remainder)

        targets = {}

        for class_name in self.class_labels:
            extra = 1 if class_name in remainder_sample else 0
            desired = min_occurrences + extra
            targets[class_name] = desired

        return targets
    
    def _random_unconstrained_targets(self) -> dict[str, int]:
        """Calculate randomized targets without capacity constraints.

        Returns
        -------
        dict[str, int]
            Randomized target count for each class.

        Raises
        ------
        ValueError
            If ``target_size`` is smaller than the number of classes.
        """
        if self.target_size < len(self.class_labels):
            raise ValueError("random distribution requires target_size >= number of classes")

        targets = {class_name: 1 for class_name in self.class_labels}
        remaining = self.target_size - sum(targets.values())

        for _ in range(remaining):
            targets[self.rng.choice(self.class_labels)] += 1

        return targets
    
    def _percentage_unconstrained_targets(self, percentages: Mapping[str, float]) -> dict[str, int]:
        """Calculate percentage-based targets without capacity constraints.

        Parameters
        ----------
        percentages : Mapping[str, float]
            Percentage allocation for each class label.

        Returns
        -------
        dict[str, int]
            Percentage-based target count for each class.

        Raises
        ------
        KeyError
            If ``percentages`` does not contain an entry for every class label.
        """
        # Targets expressed as floating point
        exact_targets: dict[str, float] = {
            class_name: self.target_size * (percentages[class_name] / 100)
            for class_name in self.class_labels
        }

        # Targets expressed as integer (cutting down decimals)
        targets: dict[str, int] = {
            class_name: int(exact_target) 
            for class_name, exact_target in exact_targets.items()
        }

        # The actual size of the dataset
        remainder = self.target_size - sum(targets.values())

        # Sorts label to have as firsts elements, the ones that exhibit an higher 
        # difference between the "exact" and "integer" dict targets
        fractional_order = sorted(
            self.class_labels,
            key=lambda class_name: exact_targets[class_name] - targets[class_name],
            reverse=True,
        )

        # Group class that belongs to the same "fraction"
        tied_groups: dict[float, list[str]] = {}
        for class_name in fractional_order:
            fraction = exact_targets[class_name] - targets[class_name]
            tied_groups.setdefault(fraction, []).append(class_name)

        # Shuffle classes in each group and build the final list
        ordered_classes: list[str] = []
        for fraction in sorted(tied_groups, reverse=True):
            class_group = tied_groups[fraction]

            # If two or more classes have the same fractional difference, those tied classes are shuffled 
            # before assigning the leftover slots, avoiding always favoring the class that appears first
            self.rng.shuffle(class_group)

            ordered_classes.extend(class_group)

        # Used the slices to guarantee the respect of target size
        for class_name in ordered_classes[:remainder]:
            targets[class_name] += 1

        return targets


class SourceDatasetDistributionCalculator(AbstractDistributionCalculator):
    """Calculate class targets constrained by source dataset capacity.

    Parameters
    ----------
    class_labels : set[str]
        Labels that should receive a target count.
    target_size : int
        Total number of instances to distribute across ``class_labels``.
    rng : random.Random
        Random number generator used to break ties and redistribute targets.
    dataset_class_count : dict[str, int]
        Available source instance count for each class label.
    redistribute_leftovers : bool, optional
        Whether to redistribute targets that were clamped by source capacity.

    Raises
    ------
    ValueError
        If ``dataset_class_count`` does not contain the same labels as
        ``class_labels``.
    """

    def __init__(
        self, 
        class_labels: set[str], 
        target_size: int, 
        rng: random.Random, 
        dataset_class_count: dict[str, int],
    ):
        super().__init__(class_labels, target_size, rng)

        if set(dataset_class_count) != set(self.class_labels):
            msg = (
                "The dataset key labels MUST correspond to the class label"
                f"{dataset_class_count.keys()} != {self.class_labels}"
            )

            raise ValueError(msg)

        self.dataset_class_count = dataset_class_count

    def _clamp_classes(self, calculated_targets: dict[str, int]) -> dict[str, int]:
        """Clamp target counts to available source capacity.

        Parameters
        ----------
        calculated_targets : dict[str, int]
            Unconstrained target count for each class.

        Returns
        -------
        dict[str, int]
            Target count for each class capped by ``dataset_class_count``.
        """
        return {
            class_name: min(target, self.dataset_class_count[class_name]) 
            for class_name, target in calculated_targets.items()
        }

    def _redistribute_instances(self, calculated_targets: dict[str, int]) -> dict[str, int]:
        """Redistribute leftover targets across classes with remaining capacity.

        Parameters
        ----------
        calculated_targets : dict[str, int]
            Target counts after clamping to source capacity.

        Returns
        -------
        dict[str, int]
            Target counts after leftover targets are redistributed.

        Raises
        ------
        ValueError
            If ``target_size`` exceeds total available source dataset capacity.
        """
        targets = calculated_targets.copy()

        classes_with_capacity_left = {
            class_name: instances - calculated_targets[class_name]
            for class_name, instances in self.dataset_class_count.items()
            if instances > calculated_targets[class_name]
        }

        total_capacity_left = sum(classes_with_capacity_left.values())
        remainder = self.target_size - sum(calculated_targets.values())

        if remainder <= 0:
            return calculated_targets

        if remainder > total_capacity_left:
            raise ValueError("target_size exceeds available source dataset capacity")

        for _ in range(remainder):
            class_names = list(classes_with_capacity_left)
            selected_class = self.rng.choices(
                population=class_names,
                weights=[classes_with_capacity_left[class_name] for class_name in class_names],
                k=1,
            )[0]

            classes_with_capacity_left[selected_class] -= 1
            targets[selected_class] += 1

            if classes_with_capacity_left[selected_class] == 0:
                del classes_with_capacity_left[selected_class]

        return targets

    def _apply_capacity_constraints(self, calculated_targets: dict[str, int]) -> dict[str, int]:
        """Apply source dataset capacity constraints to target counts.

        Parameters
        ----------
        calculated_targets : dict[str, int]
            Unconstrained target count for each class.

        Returns
        -------
        dict[str, int]
            Target count for each class after clamping and optional
            redistribution.

        Raises
        ------
        ValueError
            If redistribution is enabled and ``target_size`` exceeds total
            available source dataset capacity.
        """
        calculated_targets_updated = self._clamp_classes(calculated_targets)
        calculated_targets_updated = self._redistribute_instances(calculated_targets_updated)

        return calculated_targets_updated


class TargetDatasetDistributionCalculator(AbstractDistributionCalculator):
    """Calculate class targets for a target dataset without capacity limits.

    Parameters
    ----------
    class_labels : set[str]
        Labels that should receive a target count.
    target_size : int
        Total number of instances to distribute across ``class_labels``.
    rng : random.Random
        Random number generator used to break ties and assign random targets.
    """

    def __init__(self, class_labels: set[str], target_size: int, rng: random.Random):
        super().__init__(class_labels, target_size, rng)

    def _apply_capacity_constraints(self, calculated_targets: dict[str, int]) -> dict[str, int]:
        """Return calculated targets unchanged.

        Parameters
        ----------
        calculated_targets : dict[str, int]
            Target count for each class.

        Returns
        -------
        dict[str, int]
            The unchanged target count for each class.
        """
        # For the TargetDatasetDistributionCalculator there is no need to 
        # clamp or apply redistribution logics. This function is an identity
        return calculated_targets


def build_distribution_calculator(
    class_labels: set[str],
    target_size: int,
    rng: random.Random,
    dataset_class_count: dict[str, int] | None = None,
) -> AbstractDistributionCalculator:
    """Build the distribution calculator for source or target datasets."""
    if dataset_class_count is None:
        return TargetDatasetDistributionCalculator(
            class_labels=class_labels,
            target_size=target_size,
            rng=rng,
        )

    return SourceDatasetDistributionCalculator(
        class_labels=class_labels,
        target_size=target_size,
        rng=rng,
        dataset_class_count=dataset_class_count,
    )
