# Titanic Survival Prediction

This project is a machine learning approach to predict the survival of passengers on the Titanic based on various features such as age, gender, class, and other factors. Several models were used, including Logistic Regression, Artificial Neural Network (ANN), and Convolutional Neural Network (CNN), to evaluate their performance on the Titanic dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approaches](#modeling-approaches)
  - [Logistic Regression](#logistic-regression)
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## Project Overview
The Titanic Survival Prediction project aims to predict the survival of passengers using machine learning models. The dataset, available on Kaggle, includes various passenger details such as age, sex, ticket class, and family size. Through feature engineering, data preprocessing, and model training, we evaluated multiple models and compared their results.

### Key Highlights:
- Feature engineering to extract meaningful features from raw data.
- Data preprocessing techniques to handle missing values, encode categorical variables, and scale numerical features.
- Performance evaluation using Logistic Regression, ANN, and CNN models.

---

## Dataset Description
The dataset used is the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) dataset. It contains the following key features:

- **Survived**: Outcome of survival (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Fare**: Passenger fare
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Data Preprocessing
The preprocessing steps were critical to prepare the data for model training. These steps were consistent across all models:

- **Handling Missing Values**: Missing values in the `Age`, `Fare`, and `Embarked` columns were filled with the median and mode values, respectively.
- **Feature Engineering**: 
  - Titles were extracted from names (e.g., Mr., Mrs., Miss.).
  - A new feature `FamilySize` was created by summing `SibSp` and `Parch`.
  - An `IsAlone` feature was added, indicating whether a passenger was traveling alone.
- **Encoding Categorical Variables**: Used one-hot encoding for `Sex`, `Embarked`, and `Title` features.
- **Scaling Numerical Features**: The `Age`, `Fare`, `FamilySize`, and `IsAlone` features were scaled using `StandardScaler`.

---

## Modeling Approaches

### Logistic Regression
Logistic Regression was used as a baseline model. This model is a simple yet effective linear classification algorithm, ideal for binary classification problems like survival prediction.

#### Key Characteristics:
- **Type**: Linear Model
- **Hyperparameter Tuning**: Used GridSearchCV to optimize `C` and `penalty` hyperparameters.

#### Logistic Regression Results:
- **Validation Accuracy**: 81.75%

---

### Artificial Neural Network (ANN)
The ANN model was used to explore deeper non-linear relationships between features. It consisted of multiple layers and neurons to capture complex patterns.

#### Key Characteristics:
- **Layers**: Input layer, 2 hidden layers, output layer
- **Activation Functions**: `ReLU` for hidden layers, `sigmoid` for the output layer
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

#### ANN Results:
- **Training Accuracy**: 89.47%
- **Validation Accuracy**: 83.80%

---

### Convolutional Neural Network (CNN)
Although CNNs are typically used for image data, this project included a CNN model to experiment with feature extraction and classification for tabular data.

#### Key Characteristics:
- **Layers**: Convolutional layers followed by dense layers
- **Pooling**: MaxPooling layers to downsample the data
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

#### CNN Results:
- **Training Accuracy**: 84.97%
- **Validation Accuracy**: 80.45%

---

## Results

Here is a summary of the results for each model:

| Model                  | Training Accuracy | Validation Accuracy | Kaggle Public Score |
|------------------------|-------------------|---------------------|---------------------|
| Logistic Regression     | NaN               | 81.75%                 | 0.77272             |
| Artificial Neural Network (ANN) | 89.47%       | 83.80%               | 0.75358                 |
| Convolutional Neural Network (CNN) | 84.97%   | 80.45%               | 0.7679                 |

---

## Conclusion
Through this project, we explored different machine learning models for predicting Titanic survival. Logistic Regression served as a strong baseline, while ANN and CNN models provided slightly better results. ANN and CNN models, with their ability to capture non-linear relationships, showed promise with improved validation accuracy compared to Logistic Regression.

---

## Future Work
- **Model Improvements**: Experiment with deeper neural network architectures and advanced techniques such as ensemble methods.
- **Feature Engineering**: Additional feature engineering, including more detailed analysis of passenger relations and ticket information, could further improve the model.
- **Cross-Validation**: Implement K-fold cross-validation for more robust performance evaluation.

