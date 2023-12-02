import os
import pandas as pd
import click
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

@click.command()
@click.argument('x_train_file')
@click.argument('y_train_file')
@click.argument('output_file')
def main(x_train_file, y_train_file, output_file):
    """
    Trains a dummy classifier using training data and performs cross-validation.
    
    This function loads training features and labels from the '/results' directory,
    initializes a DummyClassifier with a fixed random state, performs cross-validation,
    and saves the cross-validation results to a CSV file in the '/results' directory.
    
    Parameters:
    x_train_file (str): Filename for the training features CSV file located in '/results'.
    y_train_file (str): Filename for the training labels CSV file located in '/results'.
    output_file (str): Filename for saving the cross-validation results CSV in '/results'.
    """
    
    # Full path for input and output files
    input_base_path = 'result/tables'
    output_base_path = 'result/tables'
    
    # Ensure the /results directory exists
    os.makedirs(output_base_path, exist_ok=True)

    # Construct full paths for input files
    full_x_train_path = os.path.join(input_base_path, x_train_file)
    full_y_train_path = os.path.join(input_base_path, y_train_file)

    # Load the datasets
    X_train = pd.read_csv(full_x_train_path)
    y_train = pd.read_csv(full_y_train_path)

    # Baseline model
    baseline = DummyClassifier(strategy='most_frequent', random_state=522)
    cv_results = cross_validate(baseline, X_train, y_train.squeeze(), cv=5, return_train_score=True)

    # Construct full path for the output file
    full_output_path = os.path.join(output_base_path, output_file)

    # Convert results to DataFrame and save to CSV in the /results directory
    pd.DataFrame(cv_results).to_csv(full_output_path, index=False)

if __name__ == "__main__":
    main()


#python scripts/baseline_model.py x_train.csv y_train.csv cv_results.csv