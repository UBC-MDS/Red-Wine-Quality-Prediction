import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split

@click.command()
@click.argument('input_file')
@click.argument('output_x_train')
@click.argument('output_x_test')
@click.argument('output_y_train')
@click.argument('output_y_test')
@click.argument('test_size', type=float)
@click.argument('random_state', type=int)
def main(input_file, output_x_train, output_x_test, output_y_train, output_y_test, test_size, random_state):
    """
    Splits a dataset into training and testing sets and saves them to specified paths.

    Parameters:
    input_file (str): Path to the input CSV file.
    output_x_train (str): Filename for the training features output.
    output_x_test (str): Filename for the testing features output.
    output_y_train (str): Filename for the training labels output.
    output_y_test (str): Filename for the testing labels output.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    This function reads a dataset from `input_file`, separates it into features and labels,
    performs a train-test split, and then saves the resulting splits into CSV files within
    the 'results/' directory. The filenames are provided as arguments.
    """
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Read in the data
    df = pd.read_csv(input_file, delimiter=';')

    # Split features and target
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save to CSV files in the results directory
    X_train.to_csv(f'results/tables/{output_x_train}', index=False)
    X_test.to_csv(f'results/tables/{output_x_test}', index=False)
    y_train.to_csv(f'results/tables/{output_y_train}', index=False)
    y_test.to_csv(f'results/tables/{output_y_test}', index=False)

if __name__ == "__main__":
    main()


