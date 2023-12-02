# Function to plot repeating histograms of all features

# Required imports
import pandas as pd
import click
import altair as alt
import io

@click.command()
@click.argument('file_path')
@click.argument('output_file')

# Define the plotting function
def main(file_path, output_file):
    """
    Plots repeated histograms to visualize the distribution of all numeric features in the dataframe.

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
    >>> python plot_repeating_hists.py ../data/winequality-red.csv ../results/figures/repeating_hists_plot.png
    """
    # Read the dataset
    df = pd.read_csv(file_path, delimiter=';')

    # Drop the 'quality' column
    feature_df = df.drop('quality', axis=1)
    feature_names = list(feature_df.columns)
    
    # Plot the repeating hists
    feature_hists = alt.Chart(feature_df).mark_bar().encode(
         alt.X(alt.repeat()).type('quantitative').bin(maxbins=40),
         y='count()',
    ).properties(
        width=250,
        height=250
    ).repeat(
        feature_names, 
        columns=3
    )

    # Plotting
    feature_hists.save(output_file, embed_options={'renderer':'png'})

if __name__ == "__main__":
    main()
