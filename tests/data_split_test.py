import sys
import os
import pytest
import click.testing
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.data_split import main  # Import your main function

# Create a test data file
def create_test_data():
    np.random.seed(42)
    data = pd.DataFrame(np.random.rand(100, 11), columns=[f'feature{i}' for i in range(10)] + ['quality'])
    data.to_csv('test_data.csv', index=False)

create_test_data()  # Create the test data before running the tests

# Test Cases
def test_basic_functionality():
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ['test_data.csv', 'x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv', 0.25, 42])
    
    assert result.exit_code == 0, f"Error: {result.output}"
    assert os.path.exists('results/tables/x_train.csv')
    assert os.path.exists('results/tables/x_test.csv')
    assert os.path.exists('results/tables/y_train.csv')
    assert os.path.exists('results/tables/y_test.csv')

def test_test_size_proportion():
    x_test = pd.read_csv('results/tables/x_test.csv')
    assert len(x_test) == 25, "Test size proportion is not correct."

def test_random_state_consistency():
    runner = click.testing.CliRunner()
    runner.invoke(main, ['test_data.csv', 'x_train_1.csv', 'x_test_1.csv', 'y_train_1.csv', 'y_test_1.csv', 0.25, 42])
    runner.invoke(main, ['test_data.csv', 'x_train_2.csv', 'x_test_2.csv', 'y_train_2.csv', 'y_test_2.csv', 0.25, 42])

    x_test_1 = pd.read_csv('results/tables/x_test_1.csv')
    x_test_2 = pd.read_csv('results/tables/x_test_2.csv')

    assert np.array_equal(x_test_1, x_test_2), "Random state consistency is not maintained."

def teardown_module(module):
    # Clean up: Remove created files and directories
    os.remove('test_data.csv')
    files_to_remove = ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv',
                       'x_train_1.csv', 'x_test_1.csv', 'y_train_1.csv', 'y_test_1.csv',
                       'x_train_2.csv', 'x_test_2.csv', 'y_train_2.csv', 'y_test_2.csv']
    for file in files_to_remove:
        if os.path.exists(os.path.join('results/tables', file)):
            os.remove(os.path.join('results/tables', file))
    if os.path.exists('results/tables'):
        os.rmdir('results/tables')
    if os.path.exists('results'):
        os.rmdir('results')


