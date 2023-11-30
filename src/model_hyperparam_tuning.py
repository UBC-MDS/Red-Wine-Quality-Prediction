from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import click
import json
import sys
import pandas as pd

@click.command()
@click.argument('x', type=str)
@click.argument('y', type=str)
@click.argument('model', type=str)
def model_hyperparam_tuning(x, y, model):
    """
    Conducts hyperparameter tuning for a model over a clean data set with only numeric features 
    and no missing data. Saves the GridSearchCV object in the results/models folder and also
    returns it.

    Parameters:
    ----------
    X : pandas.DataFrame or str
        The feature data set. Can be either a pandas data frame or a path 
        to a csv file (must be comma-delimited). Data must be all numeric
        with no missing value.
    y : pandas.DataFrame
        The response data set. Can be either a pandas data frame or a path 
        to a csv file (must be comma-delimited).
    model : str
        The model. Possible values are:
        - 'logistic': Logistic Regression.
        - 'decision_tree': Decision Tree Classifier.
        - 'knn': k-Nearest Neighbors Classifier.
        - 'svc': Support Vector Classifier.
    params : dict
        Dictionary for hyperparameters. The keys are the hyperparameter names and the values are
        lists containing the range of values.

    Returns:
    -------
    sklearn.model_selection._search.GridSearchCV
        An instance of GridSearchCV of estimator fitted on the data set.
        
    Examples:
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 522)
    >>> result = model_hyperparam_tuning(X_train, y_train, 'logistic', {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
                                                                        'class_weight': ['balanced', None]})
    >>> result.cv_results_
    
    """
    if isinstance(x, str):
        x = pd.read_csv(x)
    if isinstance(y, str):
        y = (pd.read_csv(y)).iloc[:, 0]

    models = {'logistic': LogisticRegression(random_state=522),
              'decision_tree': DecisionTreeClassifier(random_state=522),
              'knn': KNeighborsClassifier(),
              'svc': SVC(random_state=522)}

    pipe = Pipeline([('scl', StandardScaler()),
                     ('model', models[model])])
    
    #if isinstance(params, str):
     #   params = json.loads(params)
        
    param_dict = {'model__C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000], 'model__class_weight': ['balanced', None]}
    #for param in params:
    #    param_dict['model__' + param] = params[param]

    grid_search = GridSearchCV(estimator=pipe, param_grid=[param_dict], n_jobs=-1, return_train_score=True)
    
   # with open(("../results/models/" + model + "_grid_search"), 'wb') as f:
   #     pickle.dump(grid_search.fit(x, y), f)

    return grid_search.fit(x, y)

if __name__ == '__main__':
    model_hyperparam_tuning()