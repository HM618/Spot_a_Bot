import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# list of text documents
df = pd.read_csv('clean_tweets')

text = df['text']
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
