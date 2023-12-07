import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import click
import pickle

@click.command()
@click.argument('comparison_folder', type=str)
@click.argument('x_train_folder', type=str)
@click.argument('y_train_folder', type=str)
@click.argument('x_test_folder', type=str)
@click.argument('y_test_folder', type=str)
@click.argument('result_folder', type=str)
def test_set_deployment(comparison_folder, 
                        x_train_folder, y_train_folder, 
                        x_test_folder, y_test_folder, 
                        result_folder):
    """
    Uses the best model (SVC) from hyperparameter tuning on the test set, and exports the score
    as a csv file.

    Parameters:
    ----------
    comparison_folder : str
        The path to folder containing the summary file for all tables. Must end the path with a 
        slash. File name must be 'comparison_df.csv'.
    x_train_folder : str
        The path to folder containing X_train. Must end the path with a
        slash. Mind the lower case x. File name must be 'X_train.csv'.
    y_train_folder : str
        The path to folder containing y_train. Must end the path with a
        slash. File name must be 'y_train.csv'.
    x_test_folder : str
        The path to folder containing X_test. Must end the path with a
        slash. Mind the lower case x. File name must be 'X_test.csv'.
    y_test_folder : str
        The path to folder containing y_test. Must end the path with a
        slash. File name must be 'y_test.csv'.
    result_folder : str
        The path to the folder the csv is exported to. Must end the path 
        with a slash. File name is fixed to be 'test_set_score.csv'.
        
    Examples:
    --------
    >>> test_set_deployment('results/tables/', 'results/tables/', 
    >>>                     'results/tables/', 'results/tables/', 
    >>>                     'results/tables/', 'results/tables/')
    
    """
    comparison_df = pd.read_csv((comparison_folder + 'comparison_df.csv'), index_col=0)
    svc_C = comparison_df.loc['svc', 'C']
    svc_gamma = comparison_df.loc['svc', 'gamma']
    svc_class_weight = None if comparison_df.loc['svc', 'class_weight'] == 'No Class Weight' else 'balanced'

    X_train = pd.read_csv((x_train_folder + 'X_train.csv'))
    y_train = (pd.read_csv((y_train_folder + 'y_train.csv'))).iloc[:, 0]
    X_test = pd.read_csv((x_test_folder + 'x_test.csv'))
    y_test = (pd.read_csv((y_test_folder + 'y_test.csv'))).iloc[:, 0]
    
    best_pipe = make_pipeline(StandardScaler(), SVC(C=svc_C, gamma=svc_gamma, class_weight=svc_class_weight))
    best_pipe.fit(X_train, y_train)

    with open("results/models/best_pipe.pickle", 'wb') as f:
        pickle.dump(best_pipe, f)
    
    performance = pd.DataFrame({'test_set_score': [best_pipe.score(X_test, y_test)]})
    
    performance.to_csv((result_folder + 'test_set_score.csv'), index=True)

if __name__ == '__main__':
    test_set_deployment()