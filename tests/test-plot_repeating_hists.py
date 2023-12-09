# Required imports
import pandas as pd
import altair as alt
import sys
import os
import pytest
import click.testing

# Import the plot_repeating_hists function from the scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.plot_repeating_hists import main

# Check if the function executes properly without errors
def test_function_works():
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ['data/winequality-red.csv', 'results/figures/repeating_hists_plot.png'])
    
    # Check if the script ran successfully
    assert result.exit_code == 0, f"Error: {result.output} The function is not working as intended, please confirm your inputs are correct."

# Check if the output file is created
def test_output_file_created():
    runner = click.testing.CliRunner()
    output_file = 'results/figures/repeating_hists_plot.png'

    # Remove the file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Run the plot_repeating_hists function
    result = runner.invoke(main, ['data/winequality-red.csv', output_file])

    # Check if the script ran successfully
    assert result.exit_code == 0, f"Error: {result.output}"

    # Check if the output file was created
    assert os.path.exists(output_file), "Output file was not created properly."

# Check if the function handles incorrect file paths
def test_file_path():
    runner = click.testing.CliRunner()
    fake_input_file = 'fake_data.csv'
    output_file = 'results/figures/repeating_hists_plot.png'

    # Run the main function
    result = runner.invoke(main, [fake_input_file, output_file])
    
    # Check if the command-line script produced an error message
    assert "FileNotFoundError" or 'NameError' in result.output, f"Expected FileNotFoundError or NameError, but got: {result.output}"