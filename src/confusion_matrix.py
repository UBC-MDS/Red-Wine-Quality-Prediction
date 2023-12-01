# Function to create a confusion matrix

import click
import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
import pickle


@click.command()
@click.option('--model', type=click.Path(exists=True, dir_okay=False), help="path to the model file (pickle)")
@click.option('--X_test_path', type=click.Path(), help="path to X_test csv file")
@click.option('--y_test_path', type=click.Path(), help="path to y_test csv file")
@click.option('--output_file', type=str, help="path to the output SVG file")

# Define the plotting function
def main(model, X_test_path, y_test_path, output_file):
    """
    Plots a confusion matrix from 2 datasets (X_test and y_test) sourced from a csv file. 
    The output confusion matrix is saved as a SVG in the specified directory. 

    Parameters:
    ----------
    model : str
        The filepath to the pickle file containing the model.
    X_test_path : str
        The filepath to the csv containing X_test dataset.
    y_test_path : str
        The filepath to the csv containing the y_test dataset. 
    output_file : str
        The filepath for saving the output correlation matrix as a SVG image.

    Returns:
    -------
    None
        
    Examples (in cmd line):
    --------
    >>> python confusion_matrix.py --model=results/model.pickle 
    --X_test_path=../results/X_test.csv --y_test_path=../results/y_test.csv  
    --output_file=../results/correlation_matrix_plot.png
    """
    # load the model from the pickle file
    with open(model, 'rb') as model_file:
        best_pipe = pickle.load(model_file)
    
    # create the pd dataframes 
    X_test=pd.read_csv(X_test_path)
    y_test=pd.read_csv(y_test_path)

    # generat the confusion matrix plot 
    ConfusionMatrixDisplay.from_estimator(
    best_pipe,
    X_test,
    y_test,
    values_format="d"
    )

    cm_display.plot()  # do we need------------------------
    
    # Save the plot as an image file
    plt.savefig(output_file, format="svg")

if __name__ == "__main__":
    main()