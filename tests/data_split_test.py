import sys
import os
import pytest
import click.testing
import pandas as pd

# Adjust the import path to include your script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.data_split import main  # Replace with the actual directory name where your script is located

@pytest.fixture
def runner():
    return click.testing.CliRunner()

# Test if the function executes without errors for valid inputs
def test_function_execution(runner):
    with runner.isolated_filesystem():
        # Create a dummy dataset with a semicolon delimiter
        df = pd.DataFrame({'feature1': range(10), 'quality': range(10)})
        df.to_csv('test_data.csv', index=False, sep=';')

        # Create the required directory structure
        os.makedirs('results/tables', exist_ok=True)

        result = runner.invoke(main, ['test_data.csv', 'x_train.csv', 'x_test.csv', 
                                      'y_train.csv', 'y_test.csv', '0.2', '42'])

        assert result.exit_code == 0

# Test for output file creation
def test_output_file_creation(runner):
    with runner.isolated_filesystem():
        df = pd.DataFrame({'feature1': range(10), 'quality': range(10)})
        df.to_csv('test_data.csv', index=False, sep=';')

        # Create the required directory structure
        os.makedirs('results/tables', exist_ok=True)

        runner.invoke(main, ['test_data.csv', 'x_train.csv', 'x_test.csv', 
                             'y_train.csv', 'y_test.csv', '0.2', '42'])

        # Check for files in the expected directory
        assert os.path.exists('results/tables/x_train.csv')
        assert os.path.exists('results/tables/x_test.csv')
        assert os.path.exists('results/tables/y_train.csv')
        assert os.path.exists('results/tables/y_test.csv')

# Test for invalid input file
def test_invalid_input_file(runner):
    with runner.isolated_filesystem():
        # Create the required directory structure
        os.makedirs('results/tables', exist_ok=True)

        # Invoke main with a non-existent file
        result = runner.invoke(main, ['non_existent_file.csv', 'x_train.csv', 'x_test.csv', 
                                      'y_train.csv', 'y_test.csv', '0.2', '42'])

        assert result.exit_code != 0
       

# Test for incorrect test_size or random_state
def test_invalid_test_size_or_random_state(runner):
    with runner.isolated_filesystem():
        # Create a dummy dataset
        df = pd.DataFrame({'feature1': range(10), 'quality': range(10)})
        df.to_csv('test_data.csv', index=False, sep=';')

        # Create the required directory structure
        os.makedirs('results/tables', exist_ok=True)

        # Invoke main with invalid test_size and random_state
        result = runner.invoke(main, ['test_data.csv', 'x_train.csv', 'x_test.csv', 
                                      'y_train.csv', 'y_test.csv', '1.5', '-1'])

        assert result.exit_code != 0
       
