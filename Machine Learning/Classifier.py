# -*- coding: utf-8 -*-
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

# load the data
data = load_breast_cancer()

# check the type
type(data)

# this is like a dictionary where you can treat keys like attibutes
data.keys()

# 'data' (the attribute) means the input data
data.data.shape
# it has 569 samples, 30 features

# split data into train and test sets
# this lets us simulate how our model will perform in the future
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

# instantiate the classifier and train it
model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate performance
model.score(X_train, y_train)
model.score(X_test, y_test)

# how you can make predicitons
predictions = model.predict(X_test)

N = len(y_test)
np.sum(predictions == y_test)/ N

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

model = MLPClassifier()
model.fit(X_train2, y_train)

# evaluate model's performance
model.score(X_train2, y_train)
model.score(X_test2, y_test)



#model = RandomForestClassifier()
#model.fit(X, Y) # learning
#predictions = model.predict(X) # make predictions
#
#model.score(X, Y) # accuracy

