# Required imports
import pandas as pd
import altair as alt
import sys
import os
import pytest

# Import the plot_repeating_hists function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plot_repeating_hists import plot_repeating_hists

# Test data
toy_data = pd.DataFrame({'fixed_acidity': [7.3, 4.5, 3.9, 5.6] , 'volatile_acidity': [6.3, 4.6, 8.9, 9.0], 'residual_sugar': [3.2, 3.2, 3.0, 4.0], 'quality': [1, 5, 6, 5]})	

# Set-up inputs wit the toy_data and quality column
data_frame = toy_data
target_col = 'quality'

# Test that the input is a dataframe
def test_plot_repeating_hists_isdf():
    with pytest.raises(AttributeError) as excinfo:
        plot_repeating_hists('test_string', target_col)
        raise TypeError('The data_frame input must be a pandas dataframe')

# Test that target_col is in df
def test_plot_repeating_hists_target_indf():
    with pytest.raises(KeyError) as excinfo:
        plot_repeating_hists(data_frame, 12345)
        raise KeyError('The target_col must be in the dataframe')
 
# Test for correct output type
def test_plot_repeating_hists_returns_repeat_chart():
    feature_hists = plot_repeating_hists(data_frame, target_col)
    assert isinstance(feature_hists, alt.vegalite.v5.api.RepeatChart), "plot_repeating_hists` should return an Altair RepeatChart"
