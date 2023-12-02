# Function to create a correlation matrix of all features

# Required imports
import pandas as pd
import click
import matplotlib.pyplot as plt
import seaborn as sns
import io

@click.command()
@click.argument('file_path')
@click.argument('output_file')

# Define the plotting function
def main(file_path, output_file):
    """
    Plots a correlation matrix to show correlation between features. 

    Parameters:
    ----------
    file_path : str
        The filepath of the csv containing the data.
    output_file : str
        The filepath for saving the output correlation matrix as a SVG image.

    Returns:
    -------
    None
        
    Examples:
    --------
    >>> python correlation_matrix.py ../data/winequality-red.csv ../results/figures/correlation_matrix_plot.png
    """
    # Read the dataset
    df = pd.read_csv(file_path, delimiter=';')

    # Drop the 'quality' column and calculate correlation
    feature_df = df.drop('quality', axis=1)
    corr_matrix = feature_df.corr(numeric_only=True)

    # Plotting
    plt.figure(figsize=(12,14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='PiYG')

    # Save the plot as an image file
    plt.savefig(output_file)

if __name__ == "__main__":
    main()