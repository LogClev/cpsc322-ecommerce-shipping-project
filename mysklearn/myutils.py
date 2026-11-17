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