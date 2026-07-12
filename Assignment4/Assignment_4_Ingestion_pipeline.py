# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

spark = SparkSession.builder \
    .appName("Assignment_4_Data_Ingestion") \
    .getOrCreate()

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/assignment_data/adult.csv",
    header=True,
    inferSchema=True
)

display(df.limit(5))
df.printSchema() #validate the schema

# COMMAND ----------

df = df.toDF(*[
    c.replace(".", "_")
     .replace("-", "_")
     .replace(" ", "_")
    for c in df.columns
])

# COMMAND ----------

df = df.filter(col("workclass") != "?")
df = df.filter(col("occupation") != "?")
df = df.filter(col("native_country") != "?")

df = df.dropDuplicates()
df = df.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Ingestion Timestamp

# COMMAND ----------

df = df.withColumn(
    "ingestion_timestamp",
    current_timestamp()
)

# COMMAND ----------

# Add source metadata

df = df.withColumn(
    "source_file",
    lit("adult.csv")
)

# COMMAND ----------

# verifying 

display(
    df.select(
        "ingestion_timestamp",
        "source_file"
    ).limit(5)
)

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
# MAGIC ## Vector Assembler

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

# Save as Parquet

df.write.mode("overwrite").parquet(
    "/Volumes/workspace/default/assignment_data/ingested_data.parquet"
)

# COMMAND ----------

# Read Parquet

processed_df = spark.read.parquet(
    "/Volumes/workspace/default/assignment_data/ingested_data.parquet"
)

# COMMAND ----------

# Encode the label

label_indexer = StringIndexer(
    inputCol="income",
    outputCol="label"
)

processed_df = label_indexer.fit(processed_df).transform(processed_df)

# COMMAND ----------

# Train test split

train, test = processed_df.randomSplit(
    [0.8,0.2],
    seed=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree

# COMMAND ----------

dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="label"
)

model = dt.fit(train)

# COMMAND ----------

# Prediction

predictions = model.transform(test)

# COMMAND ----------

# Accuracy

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# COMMAND ----------

# Show Prediction

display(
    predictions.select(
        "label",
        "prediction",
        "probability"
    ).limit(10)
)

# COMMAND ----------

