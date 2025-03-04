# Fraud Detection Classification

## Overview
This project aims to detect fraudulent transactions using machine learning techniques. Due to the highly imbalanced nature of fraud detection datasets, we implement preprocessing steps, feature selection using Principal Component Analysis (PCA), and class balancing using Synthetic Minority Oversampling Technique (SMOTE). The models trained include **Random Forest** and **Neural Network** classifiers.

## Dataset
We use the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains anonymized transaction details with 30 numerical features.

## Steps in the Experiment
### 1. Data Preprocessing
#### Handling Missing Values
- Found missing values in `V27`, `V28`, `Amount`, and `Class`.
- Rows with missing target values (`Class`) were removed.
- Other missing values were imputed using the **median** to handle skewed distributions.

#### Target Variable Distribution
- The dataset was **highly imbalanced** (fraud ~0.24% of transactions).
- We applied **SMOTE** to create synthetic samples of fraudulent transactions, ensuring better model learning.

### 2. Dimensionality Reduction (PCA)
- **PCA** was used to reduce dimensionality while retaining 95% of the variance.
- The optimal number of components was **15**, determined using a cumulative explained variance plot.
- PCA helped improve computational efficiency and mitigate redundancy.

### 3. Train-Test Split
- The PCA-transformed dataset was split into **80% training** and **20% testing** to ensure unbiased model evaluation.

### 4. Model Training
#### Random Forest Classifier
- A robust ensemble model combining multiple decision trees to improve performance.
- Naturally handles feature importance and reduces overfitting.

#### Neural Network Classifier
- A **feed-forward Neural Network** with one hidden layer (100 nodes) trained for up to **300 iterations**.
- Suitable for capturing complex fraud patterns.

### 5. Cross-Validation
- Used **Stratified K-Fold Cross-Validation (5 splits)** to maintain class distribution across folds.
- Evaluated model performance using multiple metrics.

### 6. Evaluation Metrics
- **Accuracy**: Measures overall correct predictions.
- **Precision**: Indicates how many predicted fraud cases were actual frauds.
- **Recall**: Shows how many actual fraud cases were correctly predicted.
- **F1-score**: Balances precision and recall, crucial for imbalanced datasets.

## Results
### Model Performance
- Both models performed **exceptionally well**, achieving near-perfect evaluation metrics.
- **Neural Network** had a slightly better recall (**1.0000**), making it optimal for capturing all fraud cases.
- **Random Forest** had better precision (**0.9996**), reducing false positives.
- Despite minor differences, both models proved highly effective.
- Possible **overfitting** may be present due to the dataset's structure and balance.

## Analysis and Insights
### Handling Imbalance
- **SMOTE** effectively balanced the dataset, improving model learning.

### PCA Effectiveness
- Reduced features from **30 to 15** while retaining **95% variance**.
- Optimized training time and minimized redundancy.

### Fraud Detection Challenges
- Real-world fraud patterns evolve constantly.
- Continuous retraining and monitoring are essential for real-time fraud detection.

## Future Work
1. **Realistic Data Splits**
   - Use time-based splits (e.g., train on older data, test on newer data) for a more realistic fraud detection scenario.

2. **Hyperparameter Tuning**
   - Optimize hyperparameters for **Random Forest** (e.g., number of trees) and **Neural Network** (e.g., hidden layers, learning rate).

3. **Threshold Optimization**
   - Adjust the decision threshold to balance **precision vs. recall** based on business needs.

4. **Feature Engineering**
   - Explore additional features (e.g., transaction frequency, time of day) for better fraud detection.

## Conclusion
Both **Random Forest** and **Neural Network** models showcased excellent fraud detection capabilities. However, real-world fraud detection is more complex, requiring **continuous monitoring, adaptive models, and external data sources** to maintain accuracy over time.

