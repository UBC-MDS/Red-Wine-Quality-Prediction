#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pytest
import sys
import os

sys.path.append('..')
from scripts.model_hyperparam_tuning import model_hyperparam_tuning

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np


# In[2]:


#Test data
np.random.seed(522)
X1 = pd.DataFrame({'a': np.random.normal(0, 25, 100), 'b': np.random.normal(5, 7, 100), 'c': np.random.normal(10, 11, 100), 'd': np.random.normal(100, 500, 100)})
y1 = np.random.randint(5, size=100)

#Expected outputs from directly using sklearn functions
#Logistic Regression
model = LogisticRegression(random_state=522)

pipe = Pipeline([('scl', StandardScaler()),
                 ('model', model)])

param_dict = {'model__C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
              'model__class_weight': ['balanced', None]}

lr_grid_search = GridSearchCV(estimator=pipe, param_grid=[param_dict], n_jobs=-1, return_train_score=True)
    
lr_grid_search.fit(X1, y1)
lr_df = (pd.DataFrame(lr_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__C",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[3]:


#Expected outputs from directly using sklearn functions
#Decision Tree
model = DecisionTreeClassifier(random_state=522)

pipe = Pipeline([('scl', StandardScaler()),
                 ('model', model)])

param_dict = {'model__criterion': ['gini', 'entropy'],
              'model__max_depth': 2 ** np.arange(8),
              'model__class_weight': ['balanced', None]}

dt_grid_search = GridSearchCV(estimator=pipe, param_grid=[param_dict], n_jobs=-1, return_train_score=True)
    
dt_grid_search.fit(X1, y1)
dt_df = (pd.DataFrame(dt_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__criterion",
        "param_model__max_depth",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[4]:


#Expected outputs from directly using sklearn functions
#k-Nearest Neighbors
model = KNeighborsClassifier()

pipe = Pipeline([('scl', StandardScaler()),
                 ('model', model)])

param_dict = {'model__n_neighbors': [1, 2, 3, 4, 5, 6]}

knn_grid_search = GridSearchCV(estimator=pipe, param_grid=[param_dict], n_jobs=-1, return_train_score=True)
    
knn_grid_search.fit(X1, y1)
knn_df = (pd.DataFrame(knn_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__n_neighbors",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[5]:


#Expected outputs from directly using sklearn functions
#SVC
model = SVC(random_state=522)

pipe = Pipeline([('scl', StandardScaler()),
                 ('model', model)])

param_dict = {'model__C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
              'model__gamma': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
              'model__class_weight': ['balanced', None]}

svc_grid_search = GridSearchCV(estimator=pipe, param_grid=[param_dict], n_jobs=-1, return_train_score=True)
    
svc_grid_search.fit(X1, y1)
svc_df = (pd.DataFrame(svc_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__C",
        "param_model__gamma",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[6]:


#Outputs using model_hyperparam_tuning
#Logistic Regression
lr_func_grid_search = model_hyperparam_tuning(X1, y1, 'logistic', {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                                                   'class_weight': ['balanced', None]})

lr_func_df = (pd.DataFrame(lr_func_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__C",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[7]:


#Outputs using model_hyperparam_tuning
#Decision Tree
dt_func_grid_search = model_hyperparam_tuning(X1, y1, 'decision_tree', {'criterion': ['gini', 'entropy'],
                                                                        'max_depth': 2 ** np.arange(8),
                                                                        'class_weight': ['balanced', None]})

dt_func_df = (pd.DataFrame(dt_func_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__criterion",
        "param_model__max_depth",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[8]:


#Outputs using model_hyperparam_tuning
#k-Nearest Neighbors
knn_func_grid_search = model_hyperparam_tuning(X1, y1, 'knn', {'n_neighbors': [1, 2, 3, 4, 5, 6]})

knn_func_df = (pd.DataFrame(knn_func_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__n_neighbors",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[9]:


#Outputs using model_hyperparam_tuning
#SVC
svc_func_grid_search = model_hyperparam_tuning(X1, y1, 'svc', {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                                               'gamma': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                                               'class_weight': ['balanced', None]})

svc_func_df = (pd.DataFrame(svc_func_grid_search.cv_results_)[
    [
        "mean_test_score",
        "mean_train_score",
        "param_model__C",
        "param_model__gamma",
        "param_model__class_weight",
        "rank_test_score",
    ]
].set_index("rank_test_score").sort_index())


# In[10]:


#Test for the checking the correct best parameters
def test_model_hyperparam_tuning_returns_correct_best_params():
    assert lr_df.equals(lr_func_df)
    assert dt_df.equals(dt_func_df)
    assert knn_df.equals(knn_func_df)
    assert svc_df.equals(svc_func_df)

#Test for the correct candidates of hyperparameters
def test_model_hyperparam_tuning_correct_hyperparameters():
    #Logistic Regression
    set(pd.DataFrame(lr_func_grid_search.cv_results_)['param_model__C']) == set([0.001, 0.01, 0.1, 1.0, 10, 100, 1000])
    set(pd.DataFrame(lr_func_grid_search.cv_results_)['param_model__class_weight']) == set(['balanced', None])

    #Decision Tree
    set(pd.DataFrame(dt_func_grid_search.cv_results_)['param_model__criterion']) == set(['gini', 'entropy'])
    set(pd.DataFrame(dt_func_grid_search.cv_results_)['param_model__max_depth']) == set(2 ** np.arange(8))
    set(pd.DataFrame(dt_func_grid_search.cv_results_)['param_model__class_weight']) == set(['balanced', None])

    #k-Nearest Neighbors
    set(pd.DataFrame(knn_func_grid_search.cv_results_)['param_model__n_neighbors']) == set([1, 2, 3, 4, 5, 6])
    
    #SVC
    set(pd.DataFrame(svc_func_grid_search.cv_results_)['param_model__C']) == set([0.001, 0.01, 0.1, 1.0, 10, 100, 1000])
    set(pd.DataFrame(svc_func_grid_search.cv_results_)['param_model__gamma']) == set([0.001, 0.01, 0.1, 1.0, 10, 100, 1000])
    set(pd.DataFrame(svc_func_grid_search.cv_results_)['param_model__class_weight']) == set(['balanced', None])

#Test for possible values for model names
def test_model_hyperparam_key_error():
    with pytest.raises(KeyError, match='Select a valid model from: "logistic", "decision_tree", "knn", "svc".'):
        model_hyperparam_tuning(X1, y1, 'test', {'n_neighbors': [1, 2, 3, 4, 5, 6]})

#Test for correct hyperparameter for a model
def test_model_hyperparam_value_error():
    with pytest.raises(ValueError, match='This model does not contain one of the hyperparameters provided.'):
        model_hyperparam_tuning(X1, y1, 'knn', {'class_weight': ['balanced', None]})