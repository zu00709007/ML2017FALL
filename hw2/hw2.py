import sys
import csv
import math
import numpy
import pandas


# use [feature] to select what you want from train data
# features below use this type
# 'age':0, 'fnlwgt':0, 'education_num':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0
dic = {}
# other features choosen by number
feature = [3]
row_name = pandas.read_csv("train.csv", sep=',' , header = -1, nrows = 1).values.tolist()
row_name = row_name[0]
# start select features
for i in range(0, len(feature)):
	item = pandas.read_csv("train.csv", sep=',' , usecols=[feature[i]]).values.tolist()
	for j in range(0, len(item)):
		if item[j][0] == " ?":
			dic['?_'+row_name[feature[i]]] = '0'
			break
		else:
			dic[item[j][0]] = '0'

# start to load feature from X_train data
row_name = pandas.read_csv("X_train.csv", sep=',' , header = -1, nrows = 1).values.tolist()
row_name = row_name[0]
x_name = []
x_item = []
y_item = []
# sort the data order
for i in row_name:
	if i in dic:
		item = pandas.read_csv("X_train.csv", sep=',' , usecols=[i]).values.tolist()
		x_name.append(i)
		y_item.append(item)
print(x_name)		
# append data into x_item and y_item
for i in range(0, len(y_item[0])):
	tmp = []
	for j in range(0, len(y_item)):
		tmp += y_item[j][i]
	tmp += [1]
	x_item.append(tmp)
y_item = []
y_item = pandas.read_csv("Y_train.csv", sep=',').values.tolist()
x_item = numpy.array(x_item)
y_item = numpy.array(y_item)

# start training
training_time = 10000
learning_rate = 10
w = [0] * len(x_item[0])
x_item_t = x_item.transpose()
s_gra = numpy.zeros(len(x_item[0]))

for i in range(training_time):
	hypo = numpy.dot(x_item, w)
	loss = hypo - y_item
	cost = numpy.sum(loss**2) / len(x_item)
	print(math.sqrt(cost))
	gra = numpy.dot(x_item_t,loss)
	s_gra += gra**2
	ada = numpy.sqrt(s_gra)
	w = w - learning_rate * gra/ada
	