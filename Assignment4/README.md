# Assignment 4: Build a Data Ingestion Pipeline using PySpark

## Objective

The objective of this assignment is to build a Data Ingestion Pipeline using PySpark by reading data from a source, validating the schema, handling corrupt records, adding ingestion metadata, storing the processed data in Parquet format, and performing a simple Machine Learning task.

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

## Data Ingestion Pipeline

### Data Source

- Adult Census Income Dataset (CSV)

### Schema Validation

- Loaded the dataset using inferSchema=True
- Verified the schema using printSchema()

### Data Cleaning

- Removed missing values
- Removed duplicate records
- Dropped corrupt/null records

### Metadata Added

- Added Ingestion Timestamp using current_timestamp()
- Added Source Metadata using source_file column

### Feature Engineering

- Converted categorical variables using StringIndexer
- Created feature vectors using VectorAssembler
- Standardized features using StandardScaler

### Data Storage

- Stored the processed dataset in Parquet format.
- Reloaded the processed dataset for Machine Learning.

---

## Machine Learning

Performed a simple Machine Learning task using:

- Decision Tree Classifier

Steps:

- Encoded target variable
- Train-Test Split
- Model Training
- Prediction
- Accuracy Evaluation

---

## Files Included

```text
Assignment_4_Data_Ingestion.ipynb
Assignment_4_Data_Ingestion.py
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

## Data Ingestion Workflow

```text
CSV Dataset
      │
      ▼
Read Data
      │
      ▼
Validate Schema
      │
      ▼
Handle Missing Values
      │
      ▼
Remove Duplicate Records
      │
      ▼
Handle Corrupt Records
      │
      ▼
Add Ingestion Timestamp
      │
      ▼
Add Source Metadata
      │
      ▼
Feature Engineering
      │
      ▼
Store as Parquet
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

- Successfully built a Data Ingestion Pipeline using PySpark.
- Validated the dataset schema.
- Added ingestion timestamp and source metadata.
- Stored the processed dataset in Parquet format.
- Reloaded the processed dataset.
- Trained a Decision Tree classification model.
- Evaluated the model using classification accuracy.

---

## Conclusion

A complete Data Ingestion Pipeline was successfully implemented using PySpark. The raw dataset was ingested from a CSV source, validated, cleaned, enriched with ingestion metadata, and stored in Parquet format. The processed data was then used to train and evaluate a Decision Tree classifier, demonstrating an end-to-end data ingestion and machine learning workflow.

---

## Author

**Pari Dubey**