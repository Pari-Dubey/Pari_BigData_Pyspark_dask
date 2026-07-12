# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA

import matplotlib.pyplot as plt

# COMMAND ----------

spark = SparkSession.builder.appName("PCA_").getOrCreate()

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/assignment_data/adult.csv",
    header=True,
    inferSchema=True
)
display(df.limit(5))

# COMMAND ----------

# Rename the column because pyspark consider "." as new file inside the outer one so replacing it as "_"

df = df.toDF(*[
    c.replace(".", "_")
     .replace("-", "_")
     .replace(" ", "_")
    for c in df.columns
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handling missing values and duplicate values

# COMMAND ----------

df = df.filter(col("workclass") != "?")
df = df.filter(col("occupation") != "?")
df = df.filter(col("native_country") != "?")

df = df.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

categorical_cols = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"
]

for column in categorical_cols:

    indexer = StringIndexer(
        inputCol=column,
        outputCol=column+"_index",
        handleInvalid="keep"
    )

    df = indexer.fit(df).transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Scaling

# COMMAND ----------

feature_columns = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "workclass_index",
    "education_index",
    "marital_status_index",
    "occupation_index",
    "relationship_index",
    "race_index",
    "sex_index",
    "native_country_index"
]


assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

df = assembler.transform(df)

# COMMAND ----------

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features"
)

scaler_model = scaler.fit(df)

df = scaler_model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply PCA

# COMMAND ----------

pca = PCA(
    k=5,
    inputCol="scaled_features",
    outputCol="pca_features"
)

pca_model = pca.fit(df)

df = pca_model.transform(df)

display(df.select("pca_features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Original vs Reduced Dimensions

# COMMAND ----------

print("Original Number of Features:", len(feature_columns))
print("Reduced Number of Features:", 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explained Variance

# COMMAND ----------

print("Explained Variance:")

for i, variance in enumerate(pca_model.explainedVariance):
    print(f"Principal Component {i+1}: {variance:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Total Explained Variance

# COMMAND ----------

total_variance = float(sum(pca_model.explainedVariance))

print("Total Explained Variance:", round(total_variance,4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert PCA output into 2 columns

# COMMAND ----------

# Extracted the first two principal components (PC1 and PC2) from the PCA output for visualization.

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

get_pc1 = udf(lambda x: float(x[0]), DoubleType())
get_pc2 = udf(lambda x: float(x[1]), DoubleType())

df = df.withColumn("PC1", get_pc1("pca_features"))
df = df.withColumn("PC2", get_pc2("pca_features"))

display(df.select("PC1", "PC2").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scatter Plot

# COMMAND ----------

pdf = df.select("PC1", "PC2").toPandas()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(pdf["PC1"], pdf["PC2"], alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.show()

# COMMAND ----------

