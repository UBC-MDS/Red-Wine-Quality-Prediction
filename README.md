# Red Wine Quality Prediction

  - author: Nicole Bidwell, Alysen Townsley, Ruocong Sun, Hongyang Zhang

## About 

The aim of this project is to create a classification model to predict the quality of red wine based on its physiochemical porperties. This is a multi-class classification problem. The features are continuous and the target variable, wine quality, takes on integer values between 0 (representing poor quality) and 10 (representing high quality) (although in our data only observations between 3 and 8 were captured). K-Nearest Neighbors (KNN), Support Vector Machine with Radial Basis Function kernel (SVM RBF), Logistic Regression, and Decision Tree were all considered in finding a suitable model. This process involved hyperparameter tuning and 5-fold cross validation. In the end, the best model was SVM RBF that had a validation score of around 61 percent, and its test set performance was around 62 percent. The model was competent in predicting some of the more mediocre wines (quality of 5 or 6), but for observations who have higher or lower quality than this range the model started falling off. This suggests that this model may not be very robust against outliers and therefore the next step should be selecting and fine-tuning a model that can help identify outliers.

The dataset used in this project was accessed from UC Irvine Machine Learning Repository, found [here](https://archive.ics.uci.edu/dataset/186/wine+quality). Specifically, the red wine dataset, `winequality-red.csv`, was used. The dataset was originally referenced from the work of Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. Further details of their work can be found [here](http://www3.dsi.uminho.pt/pcortez/wine/). The features in the dataset include: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. 

# Installation

## Set up

To run this project, follow these steps.

1. Clone the repository.
``` bash
git clone https://github.com/UBC-MDS/Red-Wine-Quality-Prediction
```

2. Create the environment. In the root of the repository run:  
``` bash
conda env create --file environment.yaml
```

3. Open the analysis in Jupyter lab. In the root of the repository run:
``` bash
conda activate red_wine_quality_prediction
jupyter lab
```

4. Open `src/red_wine_quality_prediction_report.ipynb` in Jupyter Lab 

5. Under the "Kernel" tab click "Restart Kernel and Run All Cells"

## Dependencies

- `conda` (version 23.9.0 or higher)
- `nb_conda_kernels` (version 2.3.1 or higher)
- Python and packages listed in [`environment.yaml`](environment.yaml)

# License

Distributed under the MIT License. See `LISENCE` for details. 

# Contributing 

We welcome contributions! See `CONTRIBUTING.md` for details. 

## References

***[dataset reference]***
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In *Decision Support Systems*, Elsevier, 47(4):547-553, 2009.
