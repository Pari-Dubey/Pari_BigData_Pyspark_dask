# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract

# COMMAND ----------

spark = SparkSession.builder \
    .appName("Assignment_3_ETL") \
    .getOrCreate()

# COMMAND ----------

df = spark.read.csv(
    "/Volumes/workspace/default/assignment_data/adult.csv",
    header=True,
    inferSchema=True
)

display(df.limit(5))
df.printSchema()
print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform

# COMMAND ----------

df = df.toDF(*[
    c.replace(".", "_")
     .replace("-", "_")
     .replace(" ", "_")
    for c in df.columns
])

# COMMAND ----------

# Handling missing values

display(
    df.select(
        count(when(col("workclass")=="?",True)).alias("workclass"),
        count(when(col("occupation")=="?",True)).alias("occupation"),
        count(when(col("native_country")=="?",True)).alias("native_country")
    )
)

# COMMAND ----------

# Removing missing values and duplicates

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
# MAGIC ## VectorAssembler

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
# MAGIC # Load

# COMMAND ----------

df.write.mode("overwrite").parquet(
    "/Volumes/workspace/default/assignment_data/processed_data.parquet"
)

# COMMAND ----------

# load processed data

processed_df = spark.read.parquet(
    "/Volumes/workspace/default/assignment_data/processed_data.parquet"
)

# COMMAND ----------

# verify it
display(processed_df.limit(5))
print("Rows:", processed_df.count())
print("Columns:", len(processed_df.columns))

# COMMAND ----------

# Encode Target Variable

label_indexer = StringIndexer(
    inputCol="income",
    outputCol="label"
)

processed_df = label_indexer.fit(processed_df).transform(processed_df)

# COMMAND ----------

# Train test spilit

train, test = processed_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# Decision Tree Model

dt = DecisionTreeClassifier(
    featuresCol="scaled_features",
    labelCol="label"
)

model = dt.fit(train)

# COMMAND ----------

# Prediction

predictions = model.transform(test)

# COMMAND ----------

# ACCURACY 

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# COMMAND ----------

# Display Prediction 

display(
    predictions.select(
        "label",
        "prediction",
        "probability"
    ).limit(10)
)

print(model.featureImportances)

# COMMAND ----------

