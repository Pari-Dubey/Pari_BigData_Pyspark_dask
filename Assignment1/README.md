# Assignment 1: Customer Segmentation using PySpark MLlib

## Objective

The objective of this assignment is to perform customer segmentation using the K-Means clustering algorithm in PySpark MLlib.

## Dataset

**Adult Census Income Dataset**

The dataset contains demographic and income-related information such as:

- Age
- Education
- Occupation
- Marital Status
- Race
- Gender
- Working Hours
- Capital Gain
- Capital Loss
- Native Country
- Income

Dataset Size: More than 32,000 records.

## Technologies Used

- Python
- PySpark
- Databricks
- Spark MLlib
- Matplotlib

## Steps Performed

1. Imported required libraries
2. Loaded the dataset into PySpark
3. Explored the dataset
4. Handled missing values
5. Removed duplicate records
6. Performed feature engineering using StringIndexer
7. Created feature vectors using VectorAssembler
8. Standardized features using StandardScaler
9. Applied the Elbow Method to determine the optimal number of clusters
10. Trained the K-Means clustering model
11. Generated customer clusters
12. Analyzed cluster distribution

## Machine Learning Algorithm

K-Means Clustering

## Output

The Adult Census dataset was successfully segmented into six different clusters using the K-Means clustering algorithm.

## Business Insights

- Customers with similar demographic characteristics were grouped together.
- Customer segmentation can help organizations perform targeted marketing.
- Clustering helps understand customer behavior and supports data-driven decision making.

## Folder Contents

```
Assignment_1/
│
├── Assignment_1_kmeans.ipynb
├── Assignment_1_kmeans.py
├── adult.csv
├── README.md
└── screenshots/
```

## Author

Pari Dubey