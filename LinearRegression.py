import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#Titanic dataset to see who will survive given a bunch of data
#df stands for dataframe, its a specific object as part of pandas

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

#Categorical Data is data that is not numbers, categorized by something in the column
#must encode the categorical data as numbers, hence isntead of 'female' and 'male', we can do female = 0 and male = 1
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
vocabulary = [] 
for feature_name in CATEGORICAL_COLUMNS:
	vocabulary = dftrain[feature_name].unique() #gets a list of all the unique values from a given feature column
	feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
	feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#innput function, takes the pandas datafram to a TensorFlow Dataset
def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
	def input_function():
		ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df ))
		if shuffle:
			ds = ds.shuffle(1000)
		ds = ds.batch(batch_size).repeat(num_epochs)
		return ds
	return input_function

#data_df = pandas data frame
# label_df = labels, such as y_train or y_eval
# num_epochs = number of epochs (times the ai will see the training data)
# shuffle = should the data be shuffled on every epochs
# batch_size = how large is the batch that is fed to the ai at once

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) #Train
result = linear_est.evaluate(eval_input_fn)#get the model metrics/stats by testing on the testing data


clear_output()#clears the output
#print(result)#result is a dictionary

#tensor flow is more better used to predict the otucome for a batch of data not a single piece
#this is reffering to data unseen by the AI before, i.e when evaluating the model


##################################Predicting Data##############################
Prediction = list(linear_est.predict(eval_input_fn))
for i in range(0, len(Prediction)):
	print(dfeval.loc[i])
	print(Prediction[i]['probabilities'][1]*100)#List of dictionaries predicting each input data
	#chance of not suriving is 0, chance of surviving is 1
	#could loop through the dictionary and look at each person
	print(y_eval[i])
	print(F'\n')