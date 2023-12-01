import pandas as pd
import pickle
import click

@click.command()
@click.argument('folder', type=str)
def model_table_combination(folder):
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
    lr_df = pd.read_csv((folder + '/logistic_grid_search.csv'), index_col=0)
    dt_df = pd.read_csv((folder + '/decision_tree_grid_search.csv'), index_col=0)
    knn_df = pd.read_csv((folder + '/knn_grid_search.csv'), index_col=0)
    svc_df = pd.read_csv((folder + '/svc_grid_search.csv'), index_col=0)

    comparison_df = pd.concat([lr_df, dt_df, knn_df, svc_df]).sort_values('mean_test_score', ascending=False)
    comparison_df.to_csv((folder + '/comparison_df.csv'), index=True)

if __name__ == '__main__':
    model_table_combination()