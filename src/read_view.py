import pandas as pd
import os 
import pytest
import sys
from IPython.display import display

def read_view(path, separator=',', view_nrows=5): 
    """
    Reads a csv file from a directory path and displays the top (specified) 
    number of rows from the dataframe. 

    Parameters:
    -----------
    path : str
        the csv file name and path 
    separator : str, default ','
        the delimiter within the csv file
    view_nrows : int, optional, default 5
        the number of rows in the dataframe to display. Must be between 0 and 15 (inclusive). 

    Returns:
    --------
    df : pd.DataFrame
        A pandas dataframe read from the given csv file 

    Notes:
    ------
    Enter 0 for view_nrows for no display of the dataframe.  
        
    Examples:
    --------
    >>> read_veiw('data.csv')
    >>> read_veiw('data.csv', ';', 10)
    
    """
    if not 0 <= view_nrows <= 15:
        raise ValueError("Invalid value for view_nrows. view_nrows must be between 0 and 15 (inclusive).")
    
    if os.path.exists(path) == False: 
        raise AttributeError("The file does not exist in the specified location")

    df = pd.read_csv(path, sep=separator)
    
    if 0 < view_nrows <= 15: 
        display(df.head(view_nrows))

    return df 