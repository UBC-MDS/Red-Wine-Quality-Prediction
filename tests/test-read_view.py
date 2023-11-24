import pandas as pd
import pytest
import sys
import os

sys.path.append('..')
from src.read_view import read_view


import numpy as np

# test that the output dtype is correct 
def test_output_dtype():
    assert isinstance(read_view('../tests/toy_data.csv', ','), pd.DataFrame)

# test for view_nrows out of range 
def test_view_nrows_range():
    with pytest.raises(ValueError):
        read_view('../tests/toy_data.csv', ',', view_nrows=30)
    with pytest.raises(ValueError):
        read_view('../tests/toy_data.csv', ',', view_nrows=-1)

# test that the data file exists 
def test_correct_path():
    with pytest.raises(AttributeError):
        read_view('../data/does_not_exist.csv', ';')