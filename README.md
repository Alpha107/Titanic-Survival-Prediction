# Titanic Survival Prediction

## Project Overview

The Titanic Survival Prediction project aims to predict the survival of passengers aboard the Titanic based on various features such as age, gender, socio-economic class, and family dynamics. Utilizing machine learning techniques, this project explores the relationship between passenger characteristics and survival rates.

## Dataset

The project uses the Titanic dataset available on Kaggle, which consists of two main files:

- `train.csv`: Contains information about 891 passengers, including whether they survived.
- `test.csv`: Contains similar information about 418 passengers but does not disclose survival outcomes.

## Key Features

- **Passenger Name**: Extracted titles to indicate gender and social status.
- **Age**: Imputed missing values with the median based on gender and class.
- **FamilySize**: Created a feature to reflect family dynamics during the voyage.
- **IsAlone**: A binary feature indicating whether the passenger was traveling alone.
- **Fare**: Imputed missing values with the median fare.
- **Embarked**: Filled missing entries with the most common port of embarkation.

## Methodology

1. **Data Preprocessing**
   - Handled missing values appropriately.
   - Dropped the Cabin feature due to high missingness.
   - Encoded categorical variables using One-Hot Encoding.
   - Scaled numerical features to improve model performance.

2. **Modeling**
   - Used Logistic Regression as the primary model.
   - Optimized the model using GridSearchCV to fine-tune hyperparameters.
   - Applied regularization techniques to prevent overfitting.

3. **Evaluation**
   - Achieved a validation accuracy of **[0.8547]**.
  
     ![Screenshot 2024-10-16 230545](https://github.com/user-attachments/assets/dc766536-d3fb-4056-8cb0-9597ea831d81)

   - Achieved a public score of **0.77272** on Kaggle, indicating a 77.272% accuracy in predictions.

     ![Screenshot 2024-10-16 230048](https://github.com/user-attachments/assets/81af9090-a026-4753-b23d-b4b2bd85a150)


## Results

The final model was submitted to Kaggle, yielding a public score of **0.77272**. The submission file adhered to Kaggle's format requirements, including only two columns: `PassengerId` and `Survived`.

## Submission

- Link to Kaggle Submission: [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)
- Public Score: **0.77272**

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Conclusion
This project provided valuable insights into the factors affecting survival during the Titanic disaster and enhanced my skills in data preprocessing, feature engineering, and model optimization. Future improvements may involve exploring additional models or advanced techniques to further boost the prediction accuracy.

## Acknowledgments
- Kaggle Titanic Challenge for providing the dataset and challenge.
- The Kaggle community for shared resources and insights.
