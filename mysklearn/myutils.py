def get_value_counts(values: list) -> dict:
    """
    Count the frequency of each unique value in a list.

    Args:
        values (list): List of attribute values.

    Returns:
        dict: Dictionary mapping each unique value to its frequency.
    """
    value_count_dict = {}

    for value in values:
        if value in value_count_dict:
            value_count_dict[value] += 1
        else:
            value_count_dict[value] = 1

    return value_count_dict

def get_discretized_bounds(values: list, num_buckets: int = 3) -> list[float]:
    """Computes equal-width boundaries for discretizing numerical values.

    Args:
        values(list of int or float): The numerical values used to calculate
            the discretization boundaries.
        num_buckets(int): The number of buckets to divide the values into.
            Defaults to 3.

    Returns:
        bounds(list of float): The sorted boundaries separating the buckets.
            The number of boundaries is num_buckets - 1.
    """

    bounds = []
    mx = max(values)
    mn = min(values)

    rnge = mx - mn
    bucket_width = rnge / num_buckets

    curr = mn
    while len(bounds) != num_buckets - 1:
        curr += bucket_width
        bounds.append(curr)

    return bounds

def give_discretized_value(value: float, bounds: list, labels: list) -> str:
    """Assigns a categorical label to a numerical value using given boundaries.

    Args:
        value(float): The numerical value to discretize.
        bounds(list of float): The sorted boundaries separating the buckets.
        labels(list of str): The category labels corresponding to the buckets.
            Must contain one more label than the number of boundaries.

    Returns:
        label(str): The categorical label assigned to the numerical value
    """
    
    for i, bound in enumerate(bounds):
        if value < bound:
            return labels[i]

    return labels[-1]

