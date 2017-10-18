import sys
import csv
import math
import numpy
import pandas


# use [feature] to select what you want from train data
# features below use this type
# 'age':0, 'fnlwgt':0, 'education_num':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0
dic = {'age':0, 'sex':0, 'hours_per_week':0}
# other features choosen by number
feature = [1, 3, 5, 6, 7, 8]
row_name = pandas.read_csv("train.csv", sep=',' , header = -1, nrows = 1).values.tolist()
row_name = row_name[0]
# start select features
for i in range(0, len(feature)):
	item = pandas.read_csv("train.csv", sep=',' , usecols=[feature[i]]).values.tolist()
	for j in range(0, len(item)):
		if item[j][0] == " ?":
			dic['?_'+row_name[feature[i]]] = '0'
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
row_name = len(y_item[0])
# append data into x_item and y_item
for i in range(0, row_name):
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
training_time = 1111111
learning_rate = 1e-7
#w = [0] * len(x_item[0])
w = [0.018172074468535936, 0.3995235682887812, 0.0233234308587795, 0.19169441502485926, -0.4210576276299226, -0.005436692360600547, -0.3374169370704488, 0.08895952927393222, -0.6893432157526456, -0.4703316664926803, -0.03204418841904892, -0.5133023594105831, -0.7617836202492582, -0.7716953074987328, -0.31696015350235485, -0.30350363213160136, -0.5027504068443567, -0.8961805013648237, -0.6366518165738985, -0.07984696196300799, -0.05975935513360502, 0.5016486311627957, 0.7957469579682992, -0.5473240164723697, 0.7928347109294598, -0.09621390446945928, 0.9375222588212156, -0.2433616255154354, -0.5904205447142574, 0.04953440406664937, 0.6078552440400605, -0.2555145871240904, -1.203662332159091, -0.45145606499183366, -0.34461486195453966, -0.15205692856428657, -0.011529810709077106, -0.11907075671799165, 0.751507160342176, -0.8049756995831268, -0.6991575393543211, -0.5590313448541206, -1.0518065349791808, -0.13183889053384332, 0.5905281272608726, 0.2139916106377326, 0.21441418347989652, 0.39204686869722766, -0.30256013618790484, -0.5187390517711871, 0.030580652170084214, -0.3722530634379848, -0.5317170337358417, -1.253201877175671, -0.7986673007573223, 0.7369798800996499, -0.32320115263293536, -0.44789975365129236, -0.6230229987371252, -0.33192803424119716, -0.4622268035745713, -2.188278742837146]
w = numpy.array(w)[numpy.newaxis]
w = w.transpose()
x_item_t = x_item.transpose()

for i in range(0, training_time):
	inner = numpy.dot(x_item, w)
	inner = 1 / (numpy.exp(-1 * inner)+1)
	# use cross entropy to find loss
	#loss = -1 * (y_item * numpy.log(inner) + (1 - y_item) * numpy.log(1 - inner))
	#print(numpy.sum(loss))
	inner = inner - y_item
	inner = numpy.dot(x_item_t, inner)
	w = w - learning_rate * inner
print(w.transpose().tolist()[0])
