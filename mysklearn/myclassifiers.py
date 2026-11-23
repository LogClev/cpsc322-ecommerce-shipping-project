class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        # Store prior probabilities of classes in dictionary
        # Store conditionals in a nested dict

        # stores the number of instances in each class
        class_occurences = {}
        for value in y_train:
            if value in class_occurences:
                class_occurences[value] += 1
            else:
                class_occurences[value] = 1

        # Creating priors dictionary
        self.priors = class_occurences.copy()
        for key, value in self.priors.items():
            self.priors[key] = self.priors[key] / len(y_train)

        # Creating conditionals nested dict (empty to start)
        self.conditionals = {}
        for clss in class_occurences:
            self.conditionals[clss] = {}

            for i in range(len(X_train[0])):
                self.conditionals[clss][i] = {}

                for row in X_train:
                    value = row[i]
                    self.conditionals[clss][i][value] = 0

        # Update the conditionals dict with the whole counts
        for i, row in enumerate(X_train):
            curr_class = y_train[i]

            for j, attr in enumerate(row):
                self.conditionals[curr_class][j][attr] += 1

        # Change the whole counts to probs
        for clss in self.conditionals:
            clss_total = class_occurences[clss]
            for attr in self.conditionals[clss]:
                for value in self.conditionals[clss][attr]:
                    self.conditionals[clss][attr][value] = self.conditionals[clss][attr][value] / clss_total

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            # Stores the probabilities of each class for instance
            probs = []

            # key is the class, value is the actual prior prob
            for key, value in self.priors.items():
                prob = value

                # go through each instance's attributes
                for i, attr in enumerate(instance):
                    prob *= self.conditionals[key][i].get(attr, 0)

                probs.append(prob)

            mx_prob = max(probs)
            ind_mx = probs.index(mx_prob)
            predicted_key = list(self.priors.keys())[ind_mx]
            y_predicted.append(predicted_key)

        return y_predicted