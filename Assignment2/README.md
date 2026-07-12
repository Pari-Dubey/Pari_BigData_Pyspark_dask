# Dimensionality Reduction using PCA in PySpark MLlib

## Objective

The objective of this assignment is to perform dimensionality reduction on the Adult Census Income dataset using Principal Component Analysis (PCA) in PySpark MLlib. PCA helps reduce the number of features while preserving the maximum possible information (variance).

---

## Dataset

- Dataset: Adult Census Income Dataset
- Format: CSV
- Rows: 32,561
- Features: 14 Input Features

---

## Technologies Used

- Python
- PySpark
- PySpark MLlib
- Databricks
- Matplotlib

---

## Steps Performed

1. Imported required libraries.
2. Created a Spark Session.
3. Loaded the Adult Census dataset.
4. Explored the dataset.
5. Checked and handled missing values.
6. Removed duplicate records.
7. Converted categorical columns into numerical values using StringIndexer.
8. Combined all features using VectorAssembler.
9. Standardized features using StandardScaler.
10. Applied PCA with **5 Principal Components**.
11. Calculated Explained Variance.
12. Calculated Total Explained Variance.
13. Visualized the first two Principal Components using a scatter plot.
14. Compared original and reduced dimensions.

---

## Results

- Original Features: **14**
- Reduced Features: **5 Principal Components**
- Total Explained Variance: **50.81%**

PCA successfully reduced the dimensionality of the dataset while retaining approximately half of the original information.

---

## Files Included

```
Assignment_2_PCA.ipynb
Assignment_2_PCA.py
adult.csv
README.md
requirements.txt
.gitignore
```

---

## Libraries Used

```
pyspark
matplotlib
pandas
numpy
jupyter
```

---

## Conclusion

Principal Component Analysis (PCA) was successfully implemented using PySpark MLlib. The dimensionality of the dataset was reduced from 14 features to 5 principal components while preserving approximately 50.81% of the total variance. The first two principal components were visualized to observe the reduced feature space, demonstrating how PCA simplifies high-dimensional data for further analysis and machine learning tasks.

---

## Author

**Pari Dubey**