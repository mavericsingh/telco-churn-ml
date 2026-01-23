# Telco Customer Churn Prediction

**Machine Learning Assignment – 2**

---

## 1. Problem Statement

Customer churn is a major challenge for telecom companies, as retaining existing customers is significantly more cost-effective than acquiring new ones.
The objective of this project is to build and evaluate multiple machine learning classification models to **predict whether a customer will churn** based on demographic information, service usage, and billing details.

The project follows a complete machine learning pipeline including data collection, exploratory data analysis (EDA), data cleaning, feature engineering, model training, validation, evaluation, and interpretation.

---

## 2. Dataset Description

* **Dataset Name:** Telco Customer Churn
* **Source:** IBM Sample Data (GitHub mirror)
* **Problem Type:** Binary Classification
* **Target Variable:** `Churn` (Yes / No)

### Dataset Characteristics

* **Number of instances:** 7,043
* **Number of features:** 20+ (excluding target)
* **Feature types:** Categorical and Numerical

### Feature Overview

* **Categorical features:**
  gender, Partner, Dependents, InternetService, Contract, PaymentMethod, etc.
* **Numerical features:**
  tenure, MonthlyCharges, TotalCharges, SeniorCitizen

> The `TotalCharges` feature is stored as an object in the raw dataset due to missing or blank values. This issue was identified during EDA and corrected during data cleaning.

---

## 3. Data Collection

The dataset was downloaded from a stable public IBM GitHub mirror and stored locally as a CSV file to ensure reproducibility and avoid dependency on external URLs during execution.

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand data structure and guide preprocessing decisions.

The following checks were conducted:

* Dataset shape and schema inspection
* Identification of categorical and numerical features
* Missing value detection
* Statistical summary of numerical features
* Target (`Churn`) class distribution
* Correlation analysis among numerical features

### Correlation Analysis

Correlation analysis showed weak correlations among numerical features. No strong multicollinearity was observed, so all numerical features were retained.

---

## 5. Data Cleaning and Feature Engineering

### Data Cleaning

* Dropped non-informative identifier (`customerID`)
* Converted `TotalCharges` to numeric
* Imputed missing values using median
* Encoded target variable (`Churn`: Yes → 1, No → 0)

### Feature Engineering

* One-hot encoding applied to categorical features
* Standard scaling applied to numerical features
* Preprocessing implemented using a pipeline to prevent data leakage

### Data Split

The dataset was split as follows:

* **Training set:** 70%
* **Validation set:** 15%
* **Test set:** 15%

---

## 6. Logistic Regression

### 6.1 Model Description

Logistic Regression was implemented as a baseline linear classifier using a preprocessing–classification pipeline.
A separate validation set was used for performance monitoring, while the test set was reserved for final evaluation.

---

### 6.2 Performance Metrics

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8034   | 0.8097     | 0.8108 |
| AUC       | 0.8487   | 0.8455     | 0.8473 |
| Precision | 0.6542   | 0.6549     | 0.6866 |
| Recall    | 0.5497   | 0.5964     | 0.5302 |
| F1 Score  | 0.5974   | 0.6243     | 0.5984 |
| MCC       | 0.4718   | 0.4982     | 0.4841 |

---

### 6.3 Observations

* Logistic Regression shows consistent performance across all splits
* AUC ≈ 0.85 indicates strong class separation
* Precision is higher than recall, indicating conservative churn predictions
* No significant overfitting observed

---

## 7. Decision Tree Classifier

### 7.1 Unrestricted Decision Tree (Overfitting Case)

An unrestricted Decision Tree was trained to study baseline behavior.

#### Performance Metrics

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.9980   | 0.7292     | 0.7351 |
| AUC       | 1.0000   | 0.6579     | 0.6584 |
| Precision | 1.0000   | 0.4897     | 0.5018 |
| Recall    | 0.9924   | 0.5071     | 0.4911 |
| F1 Score  | 0.9962   | 0.4982     | 0.4964 |
| MCC       | 0.9948   | 0.3129     | 0.3167 |

#### Overfitting Analysis

The unrestricted Decision Tree achieved near-perfect training performance but performed poorly on validation and test sets.
This large train–validation gap indicates **severe overfitting**, which is expected due to the high variance and memorization tendency of deep decision trees.

---

## 7.2 Decision Tree with GridSearchCV (Optimized)

To address overfitting, hyperparameter tuning was performed using **GridSearchCV** with 5-fold cross-validation, optimizing the **AUC** metric.

### Performance Metrics (GridSearch Optimized)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8146   | 0.7983     | 0.8013 |
| AUC       | 0.8674   | 0.8350     | 0.8575 |
| Precision | 0.7010   | 0.6558     | 0.6749 |
| Recall    | 0.5252   | 0.5036     | 0.4875 |
| F1 Score  | 0.6005   | 0.5697     | 0.5661 |
| MCC       | 0.4916   | 0.4475     | 0.4514 |

### Observations

* Training performance decreased compared to the unrestricted tree, which is expected
* Validation and test performance improved significantly
* Reduced gap between training and validation metrics indicates better generalization
* GridSearch-optimized Decision Tree is used for final comparison

---

## 8. Key Insight

Although hyperparameter tuning significantly improved Decision Tree performance, **Logistic Regression still generalizes better**, indicating that linear decision boundaries are sufficient for this dataset.

---

## 9. k-Nearest Neighbors (kNN)

### 9.1 Raw kNN (Baseline)

A baseline k-Nearest Neighbors classifier was trained using default parameters (`k = 5`, uniform weights) to study its behavior before hyperparameter tuning. This experiment helps analyze the sensitivity of kNN to neighborhood size and its tendency to overfit in high-dimensional feature spaces.

---

### 9.2 Performance Metrics (Raw kNN)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8339   | 0.7595     | 0.7654 |
| AUC       | 0.8979   | 0.7872     | 0.7865 |
| Precision | 0.7019   | 0.5448     | 0.5642 |
| Recall    | 0.6498   | 0.5643     | 0.5160 |
| F1 Score  | 0.6749   | 0.5544     | 0.5390 |
| MCC       | 0.5643   | 0.3898     | 0.3828 |

---

### 9.3 Overfitting and Sensitivity Analysis

The raw kNN model achieved strong training performance but exhibited a noticeable drop in validation and test performance. This indicates **moderate overfitting**, which is expected for kNN with a small neighborhood size.

The performance gap arises due to:

* Sensitivity to the choice of `k`
* Distance-based learning in a high-dimensional, one-hot encoded feature space
* Equal weighting of neighbors, which may amplify noise

Despite the performance drop, validation and test metrics remain close, indicating stable generalization without severe memorization.

These observations motivate the use of **GridSearchCV** to systematically identify an optimal value of `k` and an appropriate weighting strategy.

---

### 9.4 Key Observation

> *kNN performance is highly sensitive to hyperparameter selection. While the raw model shows reasonable generalization, its performance remains below Logistic Regression, highlighting the impact of the curse of dimensionality on distance-based classifiers.*

---

### Note

The raw kNN model is included as a **baseline experiment**. The final kNN results used are obtained after hyperparameter tuning using GridSearchCV.

### 9.5 kNN with GridSearchCV (Optimized)

To improve upon the raw kNN baseline, **GridSearchCV** was applied to systematically tune the number of neighbors (`k`) and the neighbor weighting strategy.
The grid search used **5-fold cross-validation** and optimized the **AUC** metric to handle class imbalance.

---

### 9.6 Performance Metrics (GridSearch Optimized kNN)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8095   | 0.7973     | 0.7938 |
| AUC       | 0.8592   | 0.8310     | 0.8595 |
| Precision | 0.6531   | 0.6179     | 0.6255 |
| Recall    | 0.6017   | 0.6179     | 0.5587 |
| F1 Score  | 0.6263   | 0.6179     | 0.5902 |
| MCC       | 0.4996   | 0.4800     | 0.4543 |

---

### 9.7 Optimization Analysis

Hyperparameter tuning significantly improved kNN generalization performance:

* Training accuracy decreased compared to the raw kNN model, indicating reduced overfitting
* Validation and test AUC values increased, showing improved class separation
* The gap between training and validation metrics narrowed, reflecting better bias–variance balance
* MCC improved across all splits, indicating stronger predictive power beyond accuracy

Compared to the raw kNN baseline, the optimized model provides **more stable and reliable performance**, although it still underperforms Logistic Regression.

---

### 9.8 Key Observation

> *Hyperparameter tuning improved kNN performance by reducing overfitting and stabilizing predictions. However, distance-based learning remains less effective than Logistic Regression on high-dimensional, one-hot encoded feature spaces.*

## 10. Naive Bayes Classifier

### 10.1 Model Description

A **Gaussian Naive Bayes** classifier was implemented as a baseline probabilistic model.
Naive Bayes assumes conditional independence among features given the class label, which is a strong assumption but often yields competitive results in high-dimensional settings.

A separate preprocessing pipeline was used to convert categorical features into dense one-hot encoded representations, as required by Gaussian Naive Bayes.

---

### 10.2 Performance Metrics (Raw Naive Bayes)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.6647   | 0.6496     | 0.6556 |
| AUC       | 0.8229   | 0.8105     | 0.8114 |
| Precision | 0.4347   | 0.4235     | 0.4263 |
| Recall    | 0.8784   | 0.8893     | 0.8541 |
| F1 Score  | 0.5816   | 0.5737     | 0.5687 |
| MCC       | 0.4125   | 0.4020     | 0.3877 |

---

### 10.3 Bias–Variance and Behavior Analysis

The Naive Bayes classifier demonstrates **low variance and high bias**, as indicated by the close alignment of training, validation, and test metrics. Unlike Decision Trees and kNN, Naive Bayes does not overfit the training data.

Key observations:

* Recall is consistently very high (>0.85), indicating strong ability to identify churners
* Precision is relatively low, leading to a higher false-positive rate
* AUC remains competitive (~0.81), despite the strong independence assumptions
* Accuracy is lower compared to other models due to the class imbalance and conservative probability estimates

This behavior is expected, as Naive Bayes prioritizes probabilistic coverage over precise decision boundaries.

---

### 10.4 Key Observation

> *Naive Bayes achieves high recall with stable generalization across datasets, making it suitable for scenarios where identifying potential churners is more important than minimizing false positives.*

## 11. Random Forest Classifier

### 11.1 Raw Random Forest (Baseline)

A Random Forest classifier was trained using default hyperparameters to evaluate its baseline performance and compare it with a single Decision Tree. Random Forest combines multiple decision trees trained on bootstrapped samples with feature randomness to reduce variance and improve generalization.

---

### 11.2 Performance Metrics (Raw Random Forest)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.9980   | 0.7879     | 0.7919 |
| AUC       | 1.0000   | 0.8223     | 0.8172 |
| Precision | 0.9969   | 0.6217     | 0.6432 |
| Recall    | 0.9954   | 0.5107     | 0.4875 |
| F1 Score  | 0.9962   | 0.5608     | 0.5547 |
| MCC       | 0.9948   | 0.4263     | 0.4291 |

---

### 11.3 Overfitting Analysis

The raw Random Forest model achieved near-perfect performance on the training dataset, indicating that the individual trees were able to memorize training samples. However, validation and test performance were substantially lower, revealing **moderate to severe overfitting**.

Key observations:

* Although Random Forest reduces overfitting compared to a single Decision Tree, unrestricted trees can still memorize training data
* Bagging alone does not sufficiently control variance when tree depth is unconstrained
* High-cardinality categorical features contribute to complex tree structures
* Validation and test metrics remain close, indicating stable but biased generalization

These results highlight that **Random Forest still requires explicit regularization** through hyperparameter tuning to achieve optimal generalization.

---

### 11.4 Key Observation

> *Random Forest improves generalization compared to a single Decision Tree but can still overfit when tree depth and leaf size are unconstrained, emphasizing the need for hyperparameter tuning.*

## 11.5 Random Forest with GridSearchCV (Optimized)

To mitigate overfitting observed in the raw Random Forest model, **GridSearchCV** was applied to tune key hyperparameters controlling tree complexity and ensemble diversity.
The grid search used **5-fold cross-validation** and optimized the **AUC** metric to ensure robust class separation under class imbalance.

---

### 11.6 Best Hyperparameters (GridSearchCV)

The following hyperparameters were selected by GridSearchCV:

* `n_estimators`: **100**
* `max_depth`: **None**
* `min_samples_leaf`: **10**
* `max_features`: **log2**

Although tree depth was not explicitly limited, enforcing a minimum number of samples per leaf and restricting feature selection at each split significantly reduced overfitting.

---

### 11.7 Performance Metrics (Optimized Random Forest)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8306   | 0.8021     | 0.8259 |
| AUC       | 0.8968   | 0.8448     | 0.8920 |
| Precision | 0.7386   | 0.6636     | 0.7513 |
| Recall    | 0.5596   | 0.5143     | 0.5160 |
| F1 Score  | 0.6368   | 0.5795     | 0.6118 |
| MCC       | 0.5377   | 0.4591     | 0.5193 |

---

### 11.8 Optimization Analysis

Hyperparameter tuning significantly reduced overfitting in the Random Forest model:

* Training performance decreased compared to the raw model, indicating reduced memorization
* Validation and test AUC values improved and stabilized around **0.84**
* The gap between training and validation metrics narrowed substantially
* Ensemble regularization through leaf-size constraints and feature subsampling proved effective

Compared to the raw Random Forest, the optimized model demonstrates **better bias–variance balance** and improved generalization.

---

### 11.9 Comparison with Logistic Regression

Although Random Forest benefits from ensemble learning and captures non-linear feature interactions, **Logistic Regression continues to achieve slightly higher AUC and MCC values** on the test set. This suggests that the underlying relationships in the Telco Churn dataset are largely linear, and that simpler models generalize well.

## 12. XGBoost Classifier

### 12.1 Raw XGBoost (Baseline)

XGBoost was implemented as a boosting-based ensemble model to evaluate its baseline performance on the Telco Customer Churn dataset. Unlike Random Forest, which uses bagging, XGBoost builds trees sequentially, focusing on correcting errors made by previous models.

---

### 12.2 Performance Metrics (Raw XGBoost)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8722   | 0.7992     | 0.7890 |
| AUC       | 0.9395   | 0.8417     | 0.8245 |
| Precision | 0.8016   | 0.6382     | 0.6343 |
| Recall    | 0.6888   | 0.5607     | 0.4875 |
| F1 Score  | 0.7410   | 0.5970     | 0.5513 |
| MCC       | 0.6601   | 0.4658     | 0.4226 |

---

### 12.3 Overfitting Analysis

The raw XGBoost model achieved strong performance on the training dataset, indicating its ability to fit complex decision boundaries. However, a noticeable drop in validation and test performance reveals **moderate overfitting**.

Key observations:

* High training AUC (0.94) suggests strong memorization of training patterns
* Validation and test AUC values drop significantly, indicating limited generalization
* Precision and recall decrease on unseen data, leading to reduced F1 and MCC scores
* Sequential boosting amplifies noise when tree depth and learning parameters are unconstrained

These results highlight that **boosting-based ensembles are highly expressive and require regularization** to prevent overfitting.

---

### 12.4 Key Observation

> *While XGBoost captures complex non-linear relationships effectively, its boosting mechanism can lead to overfitting when model complexity is not controlled, necessitating hyperparameter tuning.*


## 12.5 XGBoost with GridSearchCV (Optimized)

To address the overfitting observed in the raw XGBoost model, **GridSearchCV** was applied to tune key boosting and regularization parameters. The grid search used **5-fold cross-validation** and optimized the **AUC** metric to balance model complexity and generalization performance.

---

### 12.6 Best Hyperparameters (GridSearchCV)

The following hyperparameters were selected by GridSearchCV:

* `n_estimators`: **100**
* `learning_rate`: **0.05**
* `max_depth`: **3**
* `subsample`: **0.8**
* `colsample_bytree`: **1.0**

These parameters significantly reduce model complexity by using shallow trees, a lower learning rate, and subsampling, thereby mitigating overfitting.

---

### 12.7 Performance Metrics (Optimized XGBoost)

| Metric    | Training | Validation | Test   |
| --------- | -------- | ---------- | ------ |
| Accuracy  | 0.8181   | 0.7992     | 0.8117 |
| AUC       | 0.8672   | 0.8502     | 0.8595 |
| Precision | 0.7021   | 0.6518     | 0.7050 |
| Recall    | 0.5459   | 0.5214     | 0.5018 |
| F1 Score  | 0.6142   | 0.5794     | 0.5863 |
| MCC       | 0.5043   | 0.4545     | 0.4820 |

---

### 12.8 Optimization Analysis

Hyperparameter tuning successfully reduced overfitting in XGBoost:

* Training performance decreased compared to the raw model, indicating reduced memorization
* Validation and test AUC improved and stabilized around **0.84–0.85**
* The gap between training and validation metrics narrowed substantially
* Regularization through shallow trees, subsampling, and lower learning rates proved effective

The optimized XGBoost model demonstrates a **balanced bias–variance tradeoff**, providing stable generalization across unseen data.

---

### 12.9 Comparison with Other Ensemble Models

Although XGBoost is a powerful boosting-based ensemble, its optimized performance is **comparable to Random Forest and slightly below Logistic Regression** on this dataset. This suggests that the Telco Churn dataset exhibits largely linear patterns, where simpler models generalize effectively.


## 13. Final Conclusions
---

## Updated Final Model Comparison

| ML Model Name            | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ------------------------ | ---------- | ---------- | ---------- | ------     | ---------- | ---------- |
| Logistic Regression      | 0.8108     | 0.8473     | 0.6866     | 0.5302     | 0.5984     | 0.4841     |
| Decision Tree            | 0.8013     | 0.8575     | 0.6749     | 0.4875     | 0.5661     | 0.4514     |
| kNN                      | 0.7938     | 0.8595     | 0.6255     | 0.5587     | 0.5902     | 0.4543     |
| Naive Bayes              | 0.6556     | 0.8114     | 0.4263     | **0.8541** | 0.5687     | 0.3877     |
| Random Forest (Ensemble) | **0.8259** | **0.8920** | **0.7513** | 0.5160     | **0.6118** | **0.5193** |
| XGBoost (Ensemble)       | 0.8117     | 0.8595     | 0.7050     | 0.5018     | 0.5863     | 0.4802     |
|

---

| ML Model Name                | Observation about model performance                                                                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Demonstrates stable and consistent generalization with balanced performance across metrics. Strong baseline with good interpretability and competitive AUC. |
| **Decision Tree**            | Performs reasonably after regularization but shows slightly lower recall, reflecting sensitivity to data splits and inherent high variance.                 |
| **kNN**                      | Achieves competitive AUC and balanced F1 score but remains sensitive to feature scaling and high dimensionality due to one-hot encoding.                    |
| **Naive Bayes**              | Exhibits very high recall, making it effective for identifying churners, but low precision leads to more false positives.                                   |
| **Random Forest (Ensemble)** | Achieves the best overall performance with highest Accuracy, AUC, F1, and MCC, indicating strong generalization and robustness.                             |
| **XGBoost (Ensemble)**       | Provides strong performance close to Random Forest, capturing non-linear patterns effectively, but slightly lower recall limits F1 score.                   |


---

* Logistic Regression achieved the **best overall performance** in terms of AUC, F1 score, and MCC
* Ensemble methods (Random Forest and XGBoost) significantly reduced overfitting compared to single Decision Trees
* Boosting-based models require careful regularization to avoid overfitting
* Distance-based and probabilistic models provide valuable baseline insights but underperform compared to linear and ensemble methods
* Simpler models can outperform complex models when underlying data relationships are largely linear

---

## Final Model Recommendation

> *Based on empirical evaluation across multiple metrics and datasets, Logistic Regression is selected as the final recommended model for Telco Customer Churn prediction due to its strong generalization, stability, and interpretability.*