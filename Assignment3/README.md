# Assignment 3: Build an ETL Pipeline using PySpark

## Objective

The objective of this assignment is to build a complete ETL (Extract, Transform, Load) pipeline using PySpark and train a Machine Learning model on the processed dataset.

---

## Dataset

- **Dataset:** Adult Census Income Dataset
- **Format:** CSV
- **Rows:** 32,561
- **Features:** 14 Input Features

---

## Technologies Used

- Python
- PySpark
- PySpark MLlib
- Databricks
- Parquet
- Decision Tree Classifier

---

## ETL Pipeline

### 1. Extract

- Loaded the Adult Census dataset from a CSV file.
- Explored the dataset structure and schema.

### 2. Transform

Performed the following preprocessing steps:

- Handled missing values
- Removed duplicate records
- Converted categorical features using StringIndexer
- Created feature vectors using VectorAssembler
- Standardized features using StandardScaler

### 3. Load

- Stored the processed dataset in **Parquet** format.
- Reloaded the processed dataset for Machine Learning.

---

## Machine Learning

After completing the ETL pipeline:

- Encoded the target variable (`income`)
- Split the dataset into training and testing sets
- Trained a Decision Tree Classifier
- Generated predictions
- Evaluated model accuracy

---

## Files Included

```text
Assignment_3_ETL.ipynb
Assignment_3_ETL.py
adult.csv
README.md
requirements.txt
.gitignore
```

---

## Libraries Used

```text
pyspark
pandas
numpy
matplotlib
jupyter
```

---

## ETL Workflow

```text
CSV Dataset
      │
      ▼
Extract
      │
      ▼
Transform
 ├── Missing Value Handling
 ├── Duplicate Removal
 ├── String Indexing
 ├── Feature Engineering
 └── Feature Scaling
      │
      ▼
Load
(Store as Parquet)
      │
      ▼
Read Processed Data
      │
      ▼
Decision Tree Classifier
      │
      ▼
Prediction & Evaluation
```

---

## Results

- Successfully built an ETL pipeline using PySpark.
- Stored the transformed dataset in Parquet format.
- Reloaded the processed dataset.
- Trained a Decision Tree classification model.
- Evaluated the model using classification accuracy.

---

## Conclusion

A complete ETL pipeline was successfully implemented using PySpark. The raw CSV dataset was extracted, transformed through preprocessing and feature engineering, and loaded into Parquet format. The processed data was then used to train and evaluate a Decision Tree classifier, demonstrating a complete end-to-end data engineering and machine learning workflow.

---

## Author

**Pari Dubey**