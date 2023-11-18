# Prediction of Red Wine Quality 

## About 

The aim of this project is to create a classification model to predict the quality of red wine based on its physiochemical porperties. This is a multi-class classification problem. The features are continuous and the target variable, wine quality, takes on integer values between 0 (representing poor quality) and 10 (representing high quality). K-Nearest Neighbors (KNN), Support Vector Machine with Radial Basis Function kernel (SVM RBF), Ridge Regression, and Linear Support Vector Classification (SVC) were all considered in finding a suitable model. ***[complete once conclusions are finalized]***

The dataset used in this project was accessed from UC Irvine Machine Learning Repository, found [here](https://archive.ics.uci.edu/dataset/186/wine+quality). Specifically, the red wine dataset, `winequality-red.csv`, was used. The dataset was originally referenced from the work of Paulo Cortez, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. Further details of their work can be found [here](http://www3.dsi.uminho.pt/pcortez/wine/). The features in the dataset include: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. 

# Installation

## Set up

To run this project, follow these steps.

1. Clone the repository.
```git clone <include link once name is changed> ```

2. Create the environment. In the root of the repository run:  
```conda env create --file environment.yml```

3. Open the analysis in Jupyter lab. In the root of the repository run: 
```conda activate <change name of env>```
```jupyter lab```

## Dependencies

***[complete once finalized]***

# License

Distributed under the MIT License. See `LISENCE` for details. 

# Contributing 

We welcome contributions! See `CONTRIBUTING.md` for details. 

## References

***[dataset reference]***
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In *Decision Support Systems*, Elsevier, 47(4):547-553, 2009.
