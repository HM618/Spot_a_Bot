import pyspark as ps
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.sql.functions import *

spark = (
        ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("lecture")
        .getOrCreate()
        )

sc = spark.sparkContext

df = spark.read.csv('clean_tweets.csv',
                         header=True)

train_df, test_df = df.randomSplit([0.7, 0.3], seed = 24)
# change label datatype from string to integer to be read by model
train_df = train_df.withColumn("label", train_df['label'].cast(IntegerType()))
# train_df.printSchema()

# get a subset of data
sample = train_df.sample(fraction=0.1)

def create_pipeline(df, model, labelCol):
    inputcols, stages = [], []
    # add different steps to stages based on data type, skip targetCol
    for col, typ in df.dtypes:
        if col == labelCol:
            continue
        elif typ == 'array<string>':
            stages.append(CountVectorizer(inputCol=col, outputCol = '{}_vec'.format(col)))
            inputcols.append('{}_vec'.format(col))
        elif typ == 'string':
            stages.append(StringIndexer(inputCol=col, outputCol = '{}_idx'.format(col), handleInvalid='skip'))
            stages.append(OneHotEncoder(inputCol='{}_idx'.format(col), outputCol = '{}_vec'.format(col)))
            inputcols.append('{}_vec'.format(col))
        else:
            inputcols.append(col)

    # assemble vectors
    stages.append(VectorAssembler(inputCols=inputcols, outputCol = 'features'))

    # add model
    stages.append(model(featuresCol='features', labelCol=labelCol))

    pipeline = Pipeline(stages = stages)

    return pipeline

#pipeline = create_pipeline(df, RandomForestRegressor, 'p2017')
print('Pipeline')
pipeline = create_pipeline(sample, RandomForestClassifier, 'features')

model = pipeline.fit(train_df)
