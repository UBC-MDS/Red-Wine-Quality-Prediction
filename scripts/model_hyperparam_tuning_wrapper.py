from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import click
import pandas as pd
import numpy as np

from model_hyperparam_tuning import model_hyperparam_tuning

@click.command()
@click.argument('x', type=str)
@click.argument('y', type=str)
@click.argument('model', type=str)
@click.argument('export_folder', type=str)
def model_hyperparam_tuning_wrapper(x, y, model, export_folder):
    """
    Tunes the hyperparameters (fixed choice and range of values) using cross-validation
    and exports in csv file the details of the best performing model with its validation
    score and hyperparameter choices. Data set is taken from the specified path.

    Parameters:
    ----------
    x : str
        The path to a csv file (must be comma-delimited). Data must be all numeric
        with no missing value.
    y : str
        The path to a csv file (must be comma-delimited). Data must be with no 
        missing value.
    model : str
        The model. Possible values are:
        - 'logistic': Logistic Regression.
        - 'decision_tree': Decision Tree Classifier.
        - 'knn': k-Nearest Neighbors Classifier.
        - 'svc': Support Vector Classifier.
    export_folder : str
        Path of where to export the csv file. Must end with a slash. When tuning multiple
        models, make sure you export all the csv files into the same folder.

    Examples:
    --------
    >>> result = model_hyperparam_tuning('../results/tables/X_train.csv', 
    >>>                                  '../results/tables/y_train.csv', 
    >>>                                  'logistic', 
    >>>                                  '../results/tables/logistic_grid_search.csv')
    
    """
    x = pd.read_csv(x)
    y = (pd.read_csv(y)).iloc[:, 0]

    param_dict = {'logistic': {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 
                               'class_weight': ['balanced', None]},
                  'decision_tree': {'criterion': ['gini', 'entropy'], 
                                    'max_depth': 2 ** np.arange(8), 
                                    'class_weight': ['balanced', None]},
                  'knn': {'n_neighbors': [1, 2, 3, 4, 5, 6]},
                  'svc': {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 
                          'gamma': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 
                          'class_weight': ['balanced', None]}}
        
    grid_search = model_hyperparam_tuning(x, y, model, param_dict[model])

    items = [
        "mean_test_score", "mean_train_score", "mean_fit_time", "rank_test_score"
    ] + [
        'param_model__' + str for str in list(param_dict[model].keys())
    ]
    
    gs_df = (pd.DataFrame(grid_search.cv_results_)[items].sort_values(
        'rank_test_score'
    ).head(1).drop(
        ['rank_test_score'], axis=1
    ).assign(model_name=model).set_index('model_name'))

    gs_df = gs_df.where(pd.notnull(gs_df), 'No Class Weight')

    gs_df.to_csv((export_folder + model + '_grid_search.csv'), index=True)

if __name__ == '__main__':
    model_hyperparam_tuning_wrapper()