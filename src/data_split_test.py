from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

import numpy as np
import pandas as pd
import pytest
import sys
import os

def splitfunction(X, y, testsize, randomstate):
    """
    Splits a dataset into training and testing sets.

    This function is a wrapper around scikit-learn's train_test_split function,
    tailored for splitting data where X and y are derived from a pandas DataFrame.

    Parameters:
    X : DataFrame
        Feature matrix, typically obtained by dropping a target column from the original DataFrame.

    y : Series
        Target variable, typically a single column from the original DataFrame.

    testsize : float or int
        If float, should be between 0.0 and 1.0 and represents the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.

    randomstate : int or RandomState instance
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns:
    X_train, X_test, y_train, y_test : arrays
        Split of the data into training and testing sets.

    Example:
    >>> df = pd.DataFrame(...) # your dataframe
    >>> X = df.drop(columns=['quality'])
    >>> y = df['quality']
    >>> X_train, X_test, y_train, y_test = splitfunction(X, y, 0.25, 42)
    """
    return train_test_split(X, y, test_size=testsize, random_state=randomstate)