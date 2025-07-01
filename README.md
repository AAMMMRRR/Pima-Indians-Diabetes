# PIMA Indians Diabetes Prediction

This project builds and evaluates multiple machine learning models to predict diabetes using the PIMA Indians Diabetes Dataset. The primary goal is to maximize recall (to minimize false negatives) and achieve high accuracy for clinical relevance. The project includes exploratory data analysis (EDA), data preprocessing, feature selection, baseline models, hyperparameter tuning, handling class imbalance with SMOTE, and ensemble stacking.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Techniques](#models-and-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The PIMA Indians Diabetes Dataset is sourced from the UCI Machine Learning Repository. It contains medical data from female patients of PIMA Indian heritage, used to predict the onset of diabetes.

- **Size**: 768 samples, 9 features (including the target variable)
- **Target Variable**: `Outcome` (0 = non-diabetic, 1 = diabetic)
- **Class Distribution**: Approximately 65% non-diabetic (0), 35% diabetic (1)

## Features
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-hour serum insulin (mu U/ml)
- **BMI**: Body mass index
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1)

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pima-indians-diabetes-prediction.git
   cd pima-indians-diabetes-prediction
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the following Python libraries:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `xgboost`
   - `optuna`
   - `imbalanced-learn`
   - `matplotlib`
   - `seaborn`

4. **Download the dataset**:
   - The dataset (`diabetes.csv`) is available from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download).
   - Place `diabetes.csv` in the project directory or update the file path in the notebook (`Pima Indians Diabetes.ipynb`).

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Pima\ Indians\ Diabetes.ipynb
   ```

2. Run the notebook cells sequentially to:
   - Load and preprocess the dataset
   - Perform exploratory data analysis (EDA)
   - Train and evaluate baseline models
   - Apply hyperparameter tuning (Grid Search, Optuna, Bayesian)
   - Handle class imbalance with SMOTE
   - Train and evaluate a stacking ensemble
   - Visualize results, including ROC curves

3. Review the final results in the "Final Results Summary" section of the notebook.

## Models and Techniques
The project implements the following machine learning models and techniques:
- **Baseline Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Naive Bayes
- **Hyperparameter Tuning**:
  - Grid Search
  - Optuna
  - Bayesian Optimization
- **Class Imbalance Handling**: Synthetic Minority Oversampling Technique (SMOTE)
- **Ensemble Method**: Stacking Classifier combining Logistic Regression, Random Forest, and XGBoost with a Logistic Regression meta-learner
- **Evaluation Metrics**:
  - Cross-Validation Accuracy
  - Test Accuracy
  - Recall (Class 1 - Diabetes)
  - ROC AUC

## Results
Key performance metrics on the test set:
- **Best Test Accuracy**: 0.7622 (Logistic Regression Baseline)
- **Best Recall (Class 1)**: 0.69 (Random Forest with SMOTE, Stacking Ensemble)
- **Best ROC AUC**: 0.75 (Random Forest with SMOTE, Stacking Ensemble with Optuna)
- **Best Cross-Validation Accuracy**: 0.7746 (Logistic Regression Baseline)

The stacking ensemble with Optuna parameters achieved a balanced performance with a test accuracy of 0.7532, recall of 0.69, and ROC AUC of 0.75, making it suitable for clinical applications where minimizing false negatives is critical.

## Limitations
- Limited to 8 features, which may restrict predictive power
- Some class overlap in the dataset
- Accuracy ceiling around 77% due to dataset characteristics
- Future improvements could include:
  - Incorporating additional features (e.g., genetics, lifestyle)
  - Advanced feature engineering
  - Neural network models
  - Alternative oversampling or ensemble techniques

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a pull request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## Repository
[https://github.com/AAMMMRRR/Pima-Indians-Diabetes]
