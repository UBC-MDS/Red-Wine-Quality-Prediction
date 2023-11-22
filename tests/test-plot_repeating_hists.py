# Required imports
import pandas as pd
import altair as alt
import sys
import os

# Import the plot_repeating_hists function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plot_repeating_hists import plot_repeating_hists

# Test data
toy_data = pd.DataFrame({'fixed_acidity': [7.3, 4.5, 3.9, 5.6] , 'volatile_acidity': [6.3, 4.6, 8.9, 9.0], 'residual_sugar': [3.2, 3.2, 3.0, 4.0], 'quality': [1, 5, 6, 5]})	

# Call the plot_repeating_hists function with the toy_data and quality column
data_frame = toy_data
target_col = 'quality'
plot_repeating_hists(data_frame, target_col)

# Test for correct target_col input type
def test_plot_repeating_hists_target():
    assert isinstance(target_col, str), 'Target column must be inputted as a string'

# Test that the input is a dataframe
def test_plot_repeating_hists_isdf():
    assert isinstance(data_frame, pd.core.frame.DataFrame), 'The data must be entered as a pandas dataframe.'

# Test that the data_frame is not empty
def test_plot_repeating_hists_is_not_empty():
    assert data_frame.empty == False, 'The data frame must not be empty' 

# Test that the target column exists in the data_frame
def test_plot_repeating_hists_target_exists():
    assert target_col in data_frame.columns, 'The specified target column must be present within the dataframe'

# Test for correct output type
def test_plot_repeating_hists_returns_repeat_chart():
    feature_hists = plot_repeating_hists(data_frame, target_col)
    assert isinstance(feature_hists, alt.vegalite.v5.api.RepeatChart), "plot_repeating_hists` should return an Altair RepeatChart"
