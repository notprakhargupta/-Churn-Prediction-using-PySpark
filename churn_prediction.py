from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnPrediction").getOrCreate()

# Load the dataset
data = spark.read.csv("customer_churn_data.csv", header=True, inferSchema=True)

# Data preprocessing - selecting features and label
features = ['tenure', 'monthly_charges', 'total_charges', 'num_of_products']
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(data)

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier(labelCol="churn", featuresCol="features", numTrees=100)
model = rf.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model using F1-score
evaluator = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"F1 Score: {f1_score:.2f}")

# Stop the Spark session
spark.stop()
