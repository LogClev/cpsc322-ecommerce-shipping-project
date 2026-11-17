import numpy as np

def random_stratified_subsample(X: list[list], y: list, test_size: float, random_state: int = 0):
    """
    Generate a random stratified train/test split while preserving class proportions.

    This function adapts scikit-learn's stratified splitting approach.
    
    Args:
        X (list[list]): Dataset containing the feature instances.
        y (list): Class labels corresponding to each instance in X.
        test_size (float): Proportion of instances to include in the test set.
        random_state (int): Random seed for reproducibility. Defaults to 0.

    Returns:
        tuple[list[int], list[int]]: A tuple containing the randomly shuffled
            training indices and testing indices, respectively.
    """
    np.random.seed(random_state)

    class_index_dict = {}
    for i, label in enumerate(y):
        if label not in class_index_dict:
            class_index_dict[label] = [i]
        else:
            class_index_dict[label].append(i)

    class_prop_dict = class_index_dict.copy()
    for label in class_prop_dict:
        class_prop_dict[label] = len(class_prop_dict[label]) / len(y)

    test_count = int(np.ceil(test_size * len(X)))
    train_count = len(y) - test_count

    train_counts_by_class = {}
    remainders_by_class = {}

    for label in class_prop_dict:
        initial_count = train_count * class_prop_dict[label]
        train_counts_by_class[label] = int(np.floor(initial_count))
        remainders_by_class[label] = initial_count - train_counts_by_class[label]

    if sum(train_counts_by_class.values()) != train_count:
        remainder = train_count - sum(train_counts_by_class.values())
        while remainder > 0:
            curr_max_clss = max(remainders_by_class, key=remainders_by_class.get)
            train_counts_by_class[curr_max_clss] += 1
            remainders_by_class[curr_max_clss] = -1
            remainder -= 1

    training_indices = []
    testing_indices = []

    for label in class_index_dict:
        shuffled_ind = class_index_dict[label].copy()
        np.random.shuffle(shuffled_ind)
        curr_train = []
        curr_test = []

        for i in range(train_counts_by_class[label]):
            curr_train.append(shuffled_ind[i])
        curr_test = shuffled_ind[train_counts_by_class[label]:]

        training_indices += curr_train
        testing_indices += curr_test

    np.random.shuffle(training_indices)
    np.random.shuffle(testing_indices)

    return training_indices, testing_indices
