import pyspark as ps
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import CountVectorizer, Tokenizer
import bleach
from pyspark.sql.functions import *

spark = (
        ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("lecture")
        .getOrCreate()
        )

sc = spark.sparkContext

# read CSV
tw_txt = spark.read.csv('clean_tweets.csv',
                         header=True)

# schema = StructType( [
#     StructField('idx',IntegerType(),True),
#     StructField('id',StringType(),True),
#     StructField('text',StringType(),True),
#     StructField('label',IntegerType(),True),
#         StructField('words',StringType(),True)] )

#tw_txt.show(5)
#type(tw_txt)

tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
tokens = tokenizer.transform(tw_txt)

# tokens.printSchema()
'''
cv = CountVectorizer(inputCol="words", outputCol="features")
cv_model = cv.fit(tokens)
cv_df = cv_model.transform(tokens)
# cv_df.show(5)
cv_df = cv_df.withColumn("label", cv_df["label"].cast(IntegerType()))
'''

OR

'''
hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(tokens)
# featurizedData.show(5)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
#rescaledData.select("label", "features").show()
'''

train_df, test_df = cv_df.randomSplit([0.7, 0.3], seed = 24)
# change label datatype from string to integer to be read by model
train_df = train_df.withColumn("label", train_df['label'].cast(IntegerType()))
# train_df.printSchema()

# get a subset of data 
sample = train_df.sample(fraction=0.1)

lr = LogisticRegression(maxIter=10)
logistic_model = lr.fit(train_df)
