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
1. Open the analysis in Jupyter lab. In the root of the repository run:
``` bash
conda activate red_wine_quality_prediction
jupyter lab
```

2. Open `notebooks/red_wine_quality_prediction_report.ipynb` in Jupyter Lab 

3. Under the "Kernel" tab click "Restart Kernel and Run All Cells"

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
