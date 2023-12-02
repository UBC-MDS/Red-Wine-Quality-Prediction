import pandas as pd
import pickle
import click

@click.command()
@click.argument('import_folder', type=str)
@click.argument('export_folder', type=str)
def model_table_combination(import_folder, export_folder):
    """
    Combines and exports as csv all the tables for the models into one big summary table.

    Parameters:
    ----------
    import_folder : str
        The path to folder to import the tables for combination. Must end the path with a 
        slash. The function assumes there are exactly four csv files all in the import_folder:
            - 'logistic_grid_search.csv': the csv file for the best logistic regression model
            - 'decision_tree_grid_search.csv': the csv file for the best decision tree model
            - 'knn_grid_search.csv': the csv file for the best knn model
            - 'svc_grid_search.csv': the csv file for the best svc model
    export_folder : str
        The path to folder to export the table as a csv file to. Must end the path with a
        slash. Recommended to be the same path as import_folder. File name is fixed to be
        'comparison_df.csv'.
        
    Examples:
    --------
    >>> model_table_combination('../results/tables/', '../results/tables/')
    
    """
    lr_df = pd.read_csv((import_folder + 'logistic_grid_search.csv'), index_col=0)
    dt_df = pd.read_csv((import_folder + 'decision_tree_grid_search.csv'), index_col=0)
    knn_df = pd.read_csv((import_folder + 'knn_grid_search.csv'), index_col=0)
    svc_df = pd.read_csv((import_folder + 'svc_grid_search.csv'), index_col=0)

    comparison_df = pd.concat([lr_df, dt_df, knn_df, svc_df]).sort_values('mean_test_score', ascending=False)
    comparison_df = comparison_df.rename(columns={'param_model__C': 'C', 'param_model__class_weight': 'class_weight',
                                                  'param_model__criterion': 'criterion', 'param_model__max_depth': 'max_depth',
                                                  'param_model__n_neighbors': 'n_neighbors', 'param_model__gamma': 'gamma'})
    comparison_df.to_csv((export_folder + 'comparison_df.csv'), index=True)

if __name__ == '__main__':
    model_table_combination()