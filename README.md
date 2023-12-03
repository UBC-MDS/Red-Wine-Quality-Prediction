# Red Wine Quality Prediction

  - author: Nicole Bidwell, Ruocong Sun, Alysen Townsley, Hongyang Zhang

## About 

The aim of this project is to create a classification model to predict the quality of red wine based on its physiochemical porperties. This is a multi-class classification problem. The features are continuous and the target variable, wine quality, takes on integer values between 0 (representing poor quality) and 10 (representing high quality) (although in our data only observations between 3 and 8 were captured). K-Nearest Neighbors (KNN), Support Vector Machine with Radial Basis Function kernel (SVM RBF), Logistic Regression, and Decision Tree were all considered in finding a suitable model. This process involved hyperparameter tuning and 5-fold cross validation. In the end, the best model was SVM RBF that had a validation score of around 61 percent, and its test set performance was around 62 percent. The model was competent in predicting some of the more mediocre wines (quality of 5 or 6), but for observations who have higher or lower quality than this range the model started falling off. This suggests that this model may not be very robust against outliers and therefore the next step should be selecting and fine-tuning a model that can help identify outliers.

The dataset used in this project was accessed from UC Irvine Machine Learning Repository, found [here](https://archive.ics.uci.edu/dataset/186/wine+quality). Specifically, the red wine dataset, `winequality-red.csv`, was used. The dataset was originally referenced from the work of Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. Further details of their work can be found [here](http://www3.dsi.uminho.pt/pcortez/wine/). The features in the dataset include: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. 

# Usage

This project can be run with either Docker (primary option) or a Virtual Environment (secondary option). Instructions for using both methods are shown below. 

## Set up

### Docker:
1. Clone the repository.
``` bash
git clone https://github.com/UBC-MDS/Red-Wine-Quality-Prediction
```

2. [Install](https://www.docker.com/get-started/) and launch Docker on your computer.


### Virtual Environment:
1. Clone the repository.
``` bash
git clone https://github.com/UBC-MDS/Red-Wine-Quality-Prediction
```

2. Create the environment. In the root of the repository run:  
``` bash
conda env create --file environment.yaml
```

## Running the Analysis

### Docker:

1. Start a new terminal session. Navigate to the root of this project repository (which you have just cloned on your local machine). Enter this command:
``` bash
docker compose up
```

2. In the terminal output you will see several URLs. Copy the one which starts with 'http://127.0.0.1:8888' and paste it into your web brower URL panel

3. Open `notebooks/red_wine_quality_prediction_report.ipynb` in Jupyter Lab 

4. Under the "Kernel" tab click "Restart Kernel and Run All Cells"


### Virtual Environment:
1. Navigate to the project root directory in `bash` and run:
``` bash
conda activate red_wine_quality_prediction
```

(Next few steps are for reproducing the report from scratch. If you do not intend to do this, skip to step 5)
2. Delete all the files in the 3 subfolders(`figures`, `models`, `tables`) of the `results` folder in the root directory. **DO NOT DELETE THESE THREE SUBFOLDERS.** You can either do it manually, or run the following command (**MAKE SURE YOU ARE CURRENTLY IN THE PROJECT ROOT FOLDER TO PREVENT ACCIDENTALLY DELETING ANYTHING.**)
``` bash
# To check your current directory. Make sure you are in the project root directory.
pwd

yes | rm results/figures/*
yes | rm results/models/*
yes | rm results/tables/*
```

3. Navigate to the `scripts` folder in the root directory in `bash`:
``` bash
# To check your current directory. If you have only been following this instruction, you should be in the correct one.
# If you are not, navigate to the project root directory.
pwd

cd scripts
```

4. Run the following commands to produce all the outputs for our report from scratch:
``` bash
# Repeating histograms for each variable in the dataset
python plot_repeating_hists.py ../data/winequality-red.csv ../results/figures/repeating_hists_plot.png

# Split data into train and test sets
python data_split.py ../data/winequality-red.csv X_train.csv X_test.csv y_train.csv y_test.csv 0.3 522

# Set dummy classifier as baseline model and return cross-validate results
python baseline_model.py X_train.csv y_train.csv cv_results.csv

# Model tunning for logistic regression model, decision tree model, KNN model and SVC model.
#Each generates a table for the details of the best performing one.
python model_hyperparam_tuning_wrapper.py ../results/tables/X_train.csv ../results/tables/y_train.csv logistic ../results/tables/
python model_hyperparam_tuning_wrapper.py ../results/tables/X_train.csv ../results/tables/y_train.csv decision_tree ../results/tables/
python model_hyperparam_tuning_wrapper.py ../results/tables/X_train.csv ../results/tables/y_train.csv knn ../results/tables/
python model_hyperparam_tuning_wrapper.py ../results/tables/X_train.csv ../results/tables/y_train.csv svc ../results/tables/

# Combine the four model tunning results into one table
python model_table_combination.py ../results/tables/ ../results/tables/

# Choose the best performance model: SVC and fit on test data 
python test_set_deployment.py ../results/tables/ ../results/tables/ ../results/tables/ ../results/tables/ ../results/tables/ ../results/tables/

# Confusion matrix for the best model SVC performance on the test data
python confusion_matrix.py --model=../results/models/best_pipe.pickle --x_test_path=../results/tables/X_test.csv --y_test_path=../results/tables/y_test.csv --output_file=../results/figures/confusion_matrix_plot.png

# Correlation matrix for all red wine physiochemical features in the data frame
python correlation_matrix.py ../data/winequality-red.csv ../results/figures/correlation_matrix_plot.png
```

5. (Once only) Make sure you are in the `red_wine_quality_prediction` environment and run:
``` bash
python -m ipykernel install --user --name conda-env-red_wine_quality_prediction-py
```

6. Navigate back to the project root directory in `bash`:
``` bash
# To check your current directory. If you have only been following this instruction, you should be in the correct one.
# If you are not, navigate to the `scripts` folder in the root directory.
pwd

cd ..
```

7. Build the `jupyter-book`:
``` bash
jupyter-book build report
```
The full report can now be viewed at the `/report/_build/html/index.html` file.

8. (Only do this and the following steps if you intend on deploying the report as a Github page)
Create a folder named `docs` (**NO OTHER NAME IS ALLOWED**) in the project root directory

9. Copy **ALL** the files in the `/report/_build/html/` directory to the `docs` folder you just created.

10. Navigate to the `docs` folder in the root directory in `bash`, and run the following:
``` bash
code .nojekyll
```
Save and close the file.

11. Push the entire project repository to Github.

12. On your repository page on Github, navigate to `Settings` $\rightarrow$ `Pages` and under `Build and deployment`, select `Deploy from a branch`. For the two dropdown tables right below it, select `main` and `/docs` for each.

13. Navigate to `Actions` tab in your repository page, and you will see that the Github page is being rendered. Once the operation is done (the yellow dot will turn green), navigate back to `Settings` $\rightarrow$ `Pages` and Github will tell you where your page is live at.

## Dependencies

Docker:
- [Docker](https://www.docker.com/) is the program used to build the container for the software used in this project. 

Virtual Environment:
- `conda` (version $\geq$ 23.9.0)
- `nb_conda_kernels` (version $\geq$ 2.3.1)
- Python and packages listed in [`environment.yaml`](environment.yaml)

# License

Distributed under the MIT License. See `LISENCE` for details. 

# Contributing 

We welcome contributions! See `CONTRIBUTING.md` for details. 

## References

***[dataset reference]***
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In *Decision Support Systems*, Elsevier, 47(4):547-553, 2009.

UCI Machine Learning Repository. (2017). https://archive.ics.uci.edu/

Red Wine Market Size, Industry Share, Trends, Analysis Forecast. (2023, January). https://www.thebusinessresearchcompany.com/report/red-wine-global-market-report#:~:text=The%20global%20red%20wine%20market%20size%20grew%20from%20%24102.97%20billion,least%20in%20the%20short%20term

Pramoditha, R. (2022, January 6). How do you apply PCA to Logistic Regression to remove Multicollinearity? Medium. https://towardsdatascience.com/how-do-you-apply-pca-to-logistic-regression-to-remove-multicollinearity-10b7f8e89f9b#:~:text=PCA%20(Principal%20Component%20Analysis)%20takes,effectively%20eliminate%20multicollinearity%20between%20features.

Deciphering interactions in logistic regression. (n.d.). https://stats.oarc.ucla.edu/stata/seminars/deciphering-interactions-in-logistic-regression/#:~:text=Logistic%20interactions%20are%20a%20complex%20concept&text=But%20in%20logistic%20regression%20interaction,can%20make%20a%20big%20difference.
