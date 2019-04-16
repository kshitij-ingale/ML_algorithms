"""
This module provides test functions for the implementation of decision tree
"""
import math
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from decision_tree import DT, read_data
from utils import accuracy, find_key_to_maxval


class TestStringMethods(unittest.TestCase):
    """
    Class to provide test cases for the implementation
    """

    def test_check_find_max_dict(self):
        """
        Test function to check if key corresponding to max value is obtained
        """
        trial_dict = {"asd": 32, "qwe": 10}
        self.assertEqual(find_key_to_maxval(trial_dict), "asd")

    def test_check_IG_with_theo(self):
        """
        Test function to validate information gain calculation with manual calculation
        """
        obj = DT()
        X = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

        def f(x):
            return math.log(x)
        h_parent = -((9 / 14) * f(9 / 14)) - ((5 / 14) * f(5 / 14))
        h_left = -((3 / 6) * f(3 / 6)) - ((3 / 6) * f(3 / 6))
        h_right = -((6 / 8) * f(6 / 8)) - ((2 / 8) * f(2 / 8))
        wt_left = 6 / 14
        wt_right = 8 / 14
        theo = h_parent - (wt_left * h_left) - (wt_right * h_right)

        self.assertEqual(obj.find_IG(X, y)[0], theo)

    def test_check_IG_with_sklearn(self):
        """
        Test function to validate information gain calculation with sklearn calculation
        """
        obj = DT()
        X = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

        sklearn_res = mutual_info_classif(np.asarray(
            X).reshape(-1, 1), np.asarray(y), discrete_features=True)[0]
        self.assertEqual(round(obj.find_IG(X, y)[0], 6), round(sklearn_res, 6))

    def test_check_small_data_performance(self):
        """
        Test function to validate implementation performance on toy dataset using sklearn classifier
        """
        # Read and encode data from csv files
        train_dir = "./data/small_train.csv"
        test_dir = "./data/small_test.csv"
        train_x, train_y, vocab, header = read_data(train_dir)
        test_x, test_y = read_data(test_dir, vocab, test_mode=True)

        # Create decision tree instance
        DT_obj = DT()

        DT_obj.train(train_x, train_y, header)

        train_predictions = DT_obj.predict(train_x, header)

        accuracy(train_y, train_predictions)
        # Obtain training accuracy using sklearn classifier
        sklearn_accuracy = get_sklearn_accuracy(mode="train")
        self.assertEqual(accuracy(train_y, train_predictions),
                         sklearn_accuracy)

        predictions = DT_obj.predict(test_x, header)
        # Obtain test accuracy using sklearn classifier
        sklearn_accuracy = get_sklearn_accuracy(mode="test")
        self.assertEqual(accuracy(test_y, predictions), sklearn_accuracy)


# Following code is mentioned in Jupyter Notebook as well

def get_sklearn_accuracy(mode):
    """
    Function to find accuracy using sklearn decision tree classifier
    Input:
    mode: variable to indicate test or train accuracy
    Output:
    accuracy score for sklearn model
    """
    if not mode:
        raise ValueError("Specify mode")
    # Read dataset
    train = pd.read_csv("data/small_train.csv")
    test = pd.read_csv("data/small_test.csv")

    # Encode categorical variables
    encoder = OrdinalEncoder()
    encoded_train = encoder.fit_transform(train)
    encoded_test = encoder.transform(test)

    # Split data into train and test dataset
    x_train, y_train = encoded_train[:, :-1], encoded_train[:, -1]
    x_test, y_test = encoded_test[:, :-1], encoded_test[:, -1]

    # Instantiate sklearn classifier and train on training dataset
    model = DecisionTreeClassifier()
    mod = model.fit(x_train, y_train)

    # Evaluate model for training dataset and test dataset accuracy
    if mode == "train":
        predictions = mod.predict(x_train)
        return accuracy_score(y_train, predictions)
    predictions = mod.predict(x_test)
    return accuracy_score(y_test, predictions)

if __name__ == '__main__':
    unittest.main()
