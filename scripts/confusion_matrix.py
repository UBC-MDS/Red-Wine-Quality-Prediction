# Script for generating confusion matrix png 

import click
import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import pickle
import pandas as pd


@click.command()
@click.option('--model', type=click.Path(exists=True, dir_okay=False), help="path to the model file (pickle)")
@click.option('--x_test_path', type=click.Path(), help="path to X_test csv file")
@click.option('--y_test_path', type=click.Path(), help="path to y_test csv file")
@click.option('--output_file', type=str, help="path to the output SVG file")

# Define the plotting function
def main(model, x_test_path, y_test_path, output_file):
    """
    Plots a confusion matrix from 2 datasets (X_test and y_test) sourced from a csv file. 
    The output confusion matrix is saved as a SVG in the specified directory. 

    Parameters:
    ----------
    model : str
        The filepath to the pickle file containing the model.
    x_test_path : str
        The filepath to the csv containing x_test dataset.
    y_test_path : str
        The filepath to the csv containing the y_test dataset. 
    output_file : str
        The filepath for saving the output confusion matrix as a png file.

    Returns:
    -------
    None
        
    Examples (in cmd line):
    --------
    >>> python confusion_matrix.py --model=../results/models/best_pipe.pickle --x_test_path=../results/tables/X_test.csv --y_test_path=../results/tables/y_test.csv --output_file=../results/figures/correlation_matrix_plot.png
    """
    # load the model from the pickle file
    with open(model, 'rb') as model_file:
        best_pipe = pickle.load(model_file)
    
    # create the pd dataframes 
    x_test=pd.read_csv(x_test_path)
    y_test=pd.read_csv(y_test_path)

    # generat the confusion matrix plot 
    ConfusionMatrixDisplay.from_estimator(
    best_pipe,
    x_test,
    y_test,
    values_format="d"
    )

    # Save the plot as an image file
    plt.savefig(output_file, format="png")

if __name__ == "__main__":
    main()