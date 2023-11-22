# function to create a repeating histogram plot to visualize the distribution of 
# all numeric features

# Required imports
import pandas as pd
import altair as alt

# Plotting function
def plot_repeating_hists(data_frame, target_col):
    """
    Plots repeated histograms to visualize the distribution of all numeric features in the dataframe. 

    Parameters:
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame containing the data to plot (containing the target column).
    target_col : str
        The name of the target column in the DataFrame (i.e. the column which we are trying to predict).

    Returns:
    -------
    alt.Chart
        An Altair.Chart class instance. 
        
    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('../data/winequality-red.csv', sep = ';')  # Replace '../data/winequality-red.csv', sep = ';' with your dataset file
    >>> feature_plot = plot_repeating_hists(data, 'quality')  # Insert target column ex/ quality.
    >>> feature_plot
    """
    # Drop the target column prior to plotting 
    feature_df = data_frame.drop(columns=target_col)

    # Extract feature names to plot
    feature_names = list(feature_df.columns)

    # Plot histograms using altair mark_bar() amd .repeat() to create a histogram for each feature. 
    feature_hists = alt.Chart(feature_df).mark_bar().encode(
         alt.X(alt.repeat()).type('quantitative').bin(maxbins=40),
         y='count()',
    ).properties(
        width=250,
        height=250
    ).repeat(
        feature_names, 
        columns=3
    ).properties(
        title = "Figure 1: Histograms showing the distrbution of each feature in the red wine dataframe."
    ).configure_title(
        orient='bottom', 
        anchor = 'middle'
    )
    return feature_hists
    