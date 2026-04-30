![](UTA-DataScience-Logo.png)

# Mushroom Classification Project

* **One Sentence Summary:** This repository uses the Mushroom Classification dataset from Kaggle to predict whether a mushroom is edible or poisonous using tabular machine learning models.

## Overview

The task is to classify mushrooms as edible or poisonous based on physical mushroom features. The dataset comes from the Kaggle Mushroom Classification dataset.

This project treats the task as a binary classification problem. I inspected the data, checked for missing and unusual values, visualized important categorical features, cleaned the data, one-hot encoded the categorical columns, trained classification models, and compared their performance.

The best models reached perfect validation and testing performance. Since perfect accuracy can look suspicious, I also checked for target leakage and row overlap between the training, validation, and testing samples.

## Summary of Workdone

### Data

* Data:
  * Type: CSV file with categorical mushroom features.
  * Input: mushroom physical features such as cap shape, cap color, odor, gill size, stalk features, population, and habitat.
  * Output: mushroom class.
    * `e` = edible
    * `p` = poisonous
  * Size: 8124 rows and 23 original columns.
  * Instances:
    * Training: 4874 rows
    * Validation: 1625 rows
    * Testing: 1625 rows

The dataset does not include a separate official Kaggle test file. Because of that, I created my own training, validation, and testing splits from the available CSV file.

#### Preprocessing / Clean up

The dataset did not contain regular missing values according to `df.isnull().sum()`. However, the `stalk-root` column contained 2480 question mark values.

These question marks were treated as an unknown category. I replaced `?` with `missing` instead of dropping the rows because removing 2480 rows would remove a large part of the dataset.

I also removed the `veil-type` column because it had only one unique value. Since every row had the same value for this column, it did not help separate edible mushrooms from poisonous mushrooms.

All feature columns were categorical, so I used one-hot encoding to convert them into numeric 0/1 columns. After encoding, the feature matrix had 116 columns.

#### Data Visualization

I made several simple visualizations to understand the data before modeling.

The class distribution showed that the dataset was close to balanced. About 51.8% of the mushrooms were edible, and about 48.2% were poisonous.

I also compared several categorical features against the target class. Odor showed a strong relationship with mushroom class. Some odor categories were mostly edible, while others were mostly poisonous. Gill size and habitat also showed useful patterns.

I also plotted the number of unique values in each column. This helped identify that `veil-type` had only one unique value, so I removed it before modeling.

### Problem Formulation

* Input:
  * One-hot encoded mushroom features.
  * These features describe physical traits such as cap shape, cap color, odor, gill size, stalk features, population, and habitat.

* Output:
  * Mushroom class.
  * `0` = edible
  * `1` = poisonous

* Models:
  * Majority-class baseline
  * Logistic Regression
  * Decision Tree
  * Random Forest

I coded poisonous mushrooms as the positive class because identifying poisonous mushrooms correctly is more important than identifying edible mushrooms correctly.

### Training

The models were trained using scikit-learn in a local Python environment.

The data was split into training, validation, and testing samples. The training sample was used to fit the models. The validation sample was used to compare model performance. The testing sample was kept separate for final evaluation.

I used `random_state=42` to make the split and models reproducible. I also used stratified splitting so that the class balance stayed similar across the training, validation, and testing samples.

I did not use deep learning for this project because the dataset is tabular, categorical, and small enough for standard scikit-learn classification models.

### Performance Comparison

The main metrics were accuracy, precision, recall, and F1 score.

Recall is important in this project because missing a poisonous mushroom would be more serious than incorrectly warning about an edible mushroom.

Validation results:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Majority Class Baseline | 0.518154 | 0.000000 | 0.000000 | 0.000000 |
| Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Decision Tree | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The majority-class baseline performed much worse than the trained models. This shows that the perfect model performance was not caused by class imbalance.

Final testing results for Random Forest:

| Metric | Result |
|---|---:|
| Accuracy | 1.000000 |
| Precision | 1.000000 |
| Recall | 1.000000 |
| F1 Score | 1.000000 |

Testing confusion matrix:

|  | Predicted Edible | Predicted Poisonous |
|---|---:|---:|
| Actual Edible | 842 | 0 |
| Actual Poisonous | 0 | 783 |

Because perfect accuracy can look suspicious, I checked for target leakage and row overlap. The target column was not included in the encoded features, and there was no overlap between the training, validation, and testing rows.

### Conclusions

The Mushroom Classification dataset is clean, categorical, and highly separable. The trained models were able to classify edible and poisonous mushrooms perfectly on both the validation and testing samples.

The main data issue was the `?` values in the `stalk-root` column. Treating those values as a `missing` category allowed the project to keep all rows instead of dropping a large part of the dataset.

The high performance should still be interpreted carefully. This result does not mean that the same model would always perform perfectly on new mushroom data collected in the real world. It mostly shows that this dataset contains strong patterns that separate edible and poisonous mushrooms.

### Future Work

Future work could include trying the model on a newer or messier mushroom dataset to see whether the performance stays high.

Another useful next step would be to compare performance after removing very strong features such as odor. This would test how much the model depends on a few highly predictive features.

A smaller model using only the most important features could also make the result easier to explain.

## How to reproduce results

### Overview of files in repository

* `README.md`: project report and summary.
* `requirements.txt`: Python packages used in the project.
* `notebooks/mushroom_classification_project.ipynb`: main notebook containing data loading, inspection, visualization, preprocessing, modeling, evaluation, and conclusion.
* `UTA-DataScience-Logo.png`: logo image used at the top of the README.

### Software Setup

This project uses Python with the following main packages:

* pandas
* numpy
* matplotlib
* scikit-learn
* kagglehub
* jupyter
* ipykernel

To set up the environment, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Data

The dataset is downloaded inside the notebook using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("uciml/mushroom-classification")
```

The original CSV file is not included in this repository. It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/uciml/mushroom-classification

### Training

Open and run the notebook:

```text
notebooks/mushroom_classification_project.ipynb
```

Run all cells from top to bottom. The notebook downloads the data, preprocesses it, trains the models, and evaluates performance.

#### Performance Evaluation

The notebook evaluates the models using:

* Accuracy
* Precision
* Recall
* F1 score
* Classification report
* Confusion matrix

The Random Forest model is used for the final held-out testing evaluation.

## Citations

* Kaggle. Mushroom Classification Dataset. https://www.kaggle.com/datasets/uciml/mushroom-classification
* UCI Machine Learning Repository. Mushroom Data Set.