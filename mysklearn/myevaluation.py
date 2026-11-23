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

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []

    for _ in range(len(labels)):
        matrix.append([0] * len(labels))

    # pylint: disable=consider-using-enumerate
    for i in range(len(y_true)):
        row_ind = labels.index(y_true[i])
        col_ind = labels.index(y_pred[i])
        matrix[row_ind][col_ind] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = 0
    num_incorrect = 0
    total = len(y_true)

    # pylint: disable=consider-using-enumerate
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            num_correct += 1
        else:
            num_incorrect += 1

    if normalize is True:
        return num_correct / total

    return num_correct

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Precision = TP / TP + FP
    # FP = false is positive label and true is predicted label

    if pos_label is None:
        if labels is not None:
            pos_label = labels[0]
        else:
            pos_label = y_true[0]

    tp = 0
    fp = 0

    for i, label in enumerate(y_true):
        if label == pos_label:
            if y_true[i] == y_pred[i]:
                tp += 1
        else:
            if y_pred[i] == pos_label:
                fp += 1

    if tp + fp == 0:
        return 0.0

    return (tp / (tp + fp))

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Recall = TP / TP + FN
    # FN = true is positive label and y_pred is smth else

    if pos_label is None:
        if labels is not None:
            pos_label = labels[0]
        else:
            pos_label = y_true[0]

    tp = 0
    fn = 0

    for i, label in enumerate(y_true):
        if label == pos_label:
            if y_true[i] == y_pred[i]:
                tp += 1
            else:
                fn += 1

    if tp + fn == 0:
        return 0.0

    return (tp / (tp + fn))

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if pos_label is None:
        if labels is not None:
            pos_label = labels[0]
        else:
            pos_label = y_true[0]

    tp = 0
    fp = 0
    fn = 0

    for i, label in enumerate(y_true):
        if label == pos_label:
            if y_true[i] == y_pred[i]:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] == pos_label:
                fp += 1

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)
