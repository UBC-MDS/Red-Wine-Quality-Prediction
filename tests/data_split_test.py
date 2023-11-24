#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pytest
import sys
import os

sys.path.append('..')
from src.data_split_test import splitfunction

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

import numpy as np


# In[2]:


#Test data
np.random.seed(522)

# Generating random data for X and y
X = np.random.rand(100, 10) # 100 samples with 10 features each
y = np.random.randint(0, 2, 100) # 100 samples with binary labels (0 or 1)


#Expected outputs from directly using sklearn functions


# Test Cases
def test_basic_functionality():
    X_train, X_test, y_train, y_test = splitfunction(X, y, 0.25, 42)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25

def test_test_size_proportion():
    _, X_test, _, _ = splitfunction(X, y, 0.30, 42)
    assert len(X_test) == 30

def test_random_state_consistency():
    _, X_test1, _, _ = splitfunction(X, y, 0.25, 42)
    _, X_test2, _, _ = splitfunction(X, y, 0.25, 42)
    assert np.array_equal(X_test1, X_test2)