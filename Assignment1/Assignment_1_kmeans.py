# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

# COMMAND ----------

spark = SparkSession.builder.appName("Customer_Clustering").getOrCreate()

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/assignment_data/adult.csv",
    header=True,
    inferSchema=True
)
display(df.limit(5))

# COMMAND ----------

df.printSchema()
print("Rows :", df.count())
print("Columns :", len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing

# COMMAND ----------

# replacing the dot in the column names
df = df.toDF(*[
    c.replace(".", "_")
     .replace("-", "_")
     .replace(" ", "_")
    for c in df.columns
])
print(df.columns)

# COMMAND ----------

# checking the missing values 
 
display(
    df.select(
        count(when(col("workclass") == "?", True)).alias("workclass_missing"),
        count(when(col("occupation") == "?", True)).alias("occupation_missing"),
        count(when(col("native_country") == "?", True)).alias("native_country_missing")
    )
)

# COMMAND ----------

# Removing the missing values

df = df.filter(col("workclass") != "?")
df = df.filter(col("occupation") != "?")
df = df.filter(col("native_country") != "?")

# COMMAND ----------

# verifying it 

display(
    df.select(
        count(when(col("workclass") == "?", True)).alias("workclass_missing"),
        count(when(col("occupation") == "?", True)).alias("occupation_missing"),
        count(when(col("native_country") == "?", True)).alias("native_country_missing")
    )
)

# COMMAND ----------

# Remove Duplicates rows if exists

print("Rows Before:", df.count())
df = df.dropDuplicates()
print("Rows After:", df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Categorical columns 

categorical_cols = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"]

for column in categorical_cols:

    indexer = StringIndexer(
        inputCol=column,
        outputCol=column + "_index",
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
display(df.limit(5).select("features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardize the feature

# COMMAND ----------

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(df)
df = scaler_model.transform(df)
display(df.select("scaled_features").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Optimal K (Elbow Method)

# COMMAND ----------

cost = []

for k in range(2,11):

    kmeans = KMeans(
        featuresCol="scaled_features",
        k=k,
        seed=42
    )

    model = kmeans.fit(df)

    cost.append(model.summary.trainingCost)

# COMMAND ----------

plt.figure(figsize=(8,5))
plt.plot(range(2,11), cost, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WSSE")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# COMMAND ----------

# we will choose k = 6 that is pretty stable than the other bends (both balanced and showing the significant drop)

# COMMAND ----------

# MAGIC %md
# MAGIC ## K-Means Clustering

# COMMAND ----------

kmeans = KMeans(
    featuresCol="scaled_features",
    predictionCol="cluster",
    k=6,
    seed=42
)

model = kmeans.fit(df)

df = model.transform(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Cluster Assignment

# COMMAND ----------

display(
    df.select("cluster").limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster Distribution

# COMMAND ----------

display(
    df.groupBy("cluster")
      .count()
      .orderBy("cluster")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Cluster Centers

# COMMAND ----------

centers = model.clusterCenters()

for i, center in enumerate(centers):
    print(f"\nCluster {i}")
    print(center)

# COMMAND ----------

