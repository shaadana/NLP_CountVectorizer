"""This code classifies emotion using KNeighborsClassifier (KNN) and CountVectorizer"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_extraction.text import CountVectorizer

################
#TRANSFORMATION# text to numbers
################

#Step 1: Data
corpus = ["annoying", "awful", "boring", "very annoying annoying", "funny", "interesting"]
review_test_data = ["strange", "funny weird", "awesome", "predictable"]
y_train = ["bad", "bad", "bad", "bad", "good", "good"]
y_test_key = ["bad", "good", "good", "bad"]

#Step 2: Model (Pre-Processing)
vectorizer = CountVectorizer()

#Step 3: Training
vectorizer.fit(corpus)

#Step 4: Transforming
sparse_matrix_train = vectorizer.transform(corpus)
X_train = sparse_matrix_train.toarray()

sparse_matrix_test = vectorizer.transform(review_test_data)
X_test = sparse_matrix_test.toarray()

##################
#MACHINE LEARNING# predicting the emotions
##################

#Step 1: Data (done with data transformation lines 10-14 and 22-27)

#Step 2: Algorithm
model = KNN(n_neighbors = 1)

#Step 3: Learning/Training
model.fit(X_train, y_train)

#Step 4: Prediction/Testing
test_prediction = model.predict(X_test)
print(test_prediction)
print(vectorizer.vocabulary_)
print(X_train)
print(X_test)
print(y_train)
