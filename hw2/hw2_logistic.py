import sys
import csv
import math
import numpy
import pandas

# below is testing code
features = ['age', 'fnlwgt', 'sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
w = [15.934917582125511, 1.4081119841209881, 0.8755359674051237, 32.475697351869435, 2.84916117971882, 0.2281635681985599, -0.4408718725267887, -0.8920373634206769, -0.20029516847374618, 0.10679117207843328, -0.636754589175673, -0.6022819850159132, -3.6758707947194122, 0.05278888908608706, -0.6021804017197666, -0.5640643713736095, -0.18889515608970286, -1.4229956762684441, -1.0997684898221538, -1.0676865542657659, -0.8692240115334101, 0.6208465014354584, 0.630228719383042, 1.2331547171831598, 2.345803136143706, 0.13291793074292196, 1.523016178624274, -9.386357862475744, 2.170897877519711, 0.4839393185472818, -0.08616736617277507, -1.1719322698892733, -0.030576709187627296, 0.7957866854171967, -0.7829372584338853, -0.7678664704683746, -0.40687541373710123, -0.9336434154630938, -3.6913976884685327, 0.46682816662199345, 0.6536354210174239, 0.3029892725651639, 0.5375552491289959, -0.10651787256454869, -0.8392484743345918, 0.1818444397467347, -1.6002842479509647, -1.7203741690940828, -2.5702261866472282, -1.7490236702207917, 1.3976956901971234, -1.4404948652607192, -1.0293977275299782, -1.1407698959805268, -1.5855812586095053, -0.8641243965884222, -6.676297897161008, -1.5440749646770928, -8.774081036112685, -6.060368143969268]
x_item = []
y_item = []
# sort the data order
for i in features:
	x_item.append(pandas.read_csv(sys.argv[5], sep=',' , usecols=[i]).values.tolist())

# data normalization
tmp = numpy.array(x_item[0])
tmp = tmp / 90
x_item[0] = tmp.tolist()
tmp = numpy.array(x_item[1])
tmp = tmp / 1484705
x_item[1] = tmp.tolist()
tmp = numpy.array(x_item[3])
tmp = tmp / 99999
x_item[3] = tmp.tolist()
tmp = numpy.array(x_item[4])
tmp = tmp / 4356
x_item[4] = tmp.tolist()

# square item
row_name = [0, 1]
for i in row_name:
	tmp = numpy.array(x_item[i])
	tmp = tmp ** 2
	x_item.append(tmp.tolist())

# 3rd power
row_name = [0]
for i in row_name:
	tmp = numpy.array(x_item[i])
	tmp = tmp ** 3
	x_item.append(tmp.tolist())

# append data into x_item and y_item
for i in range(0, len(x_item[0])):
	tmp = []
	for j in range(0, len(x_item)):
		tmp += x_item[j][i]
	tmp += [1]
	y_item.append(tmp)

x_item = numpy.array(y_item)
y_item = numpy.dot(x_item, w)
y_item = 1 / (1 + numpy.exp(-1 * y_item))

index = 1
with open(sys.argv[6], 'w', encoding='utf-8') as f:
	spamwriter = csv.writer(f, delimiter=',')
	spamwriter.writerow(['id', 'label'])
	# start testing x_vector to get y
	for i in y_item:
		if i < 0.5:
			spamwriter.writerow([str(index), 0])
		else:
			spamwriter.writerow([str(index), 1])
		index += 1
		
# below is training code
"""
# use [feature] to select what you want from train data
# features below use this type
# 'age':0, 'fnlwgt':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0
dic = {'age':0, 'sex':0, 'capital_gain':0, 'capital_loss':0}
# other features choosen by number
feature = [1, 3, 5, 6, 7, 8, 13]
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

# data normalization
row_name = [0, 2, 3]
for i in row_name:
	tmp = numpy.array(y_item[i])
	print(numpy.amax(tmp))
	tmp = tmp / numpy.amax(tmp)
	y_item[i] = tmp.tolist()
	
# square item
row_name = [0]
for i in row_name:
	tmp = numpy.array(y_item[i])
	tmp = tmp ** 2
	y_item.append(tmp.tolist())

# 3rd power
row_name = [0]
for i in row_name:
	tmp = numpy.array(y_item[i])
	tmp = tmp ** 3
	y_item.append(tmp.tolist())
	
# append data into x_item and y_item
row_name = len(y_item[0])
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
training_time = 45678
learning_rate = 1e-4
w = [0] * len(x_item[0])
w = numpy.array(w)[numpy.newaxis]
w = w.transpose()
x_item_t = x_item.transpose()

for i in range(0, training_time):
	inner = numpy.dot(x_item, w)
	inner = 1 / (1 + numpy.exp(-1 * inner))
	# use cross entropy to find loss
	#loss = -1 * (y_item * numpy.log(inner) + (1 - y_item) * numpy.log(1 - inner))
	#print(numpy.sum(loss))
	inner = inner - y_item
	inner = numpy.dot(x_item_t, inner)
	w = w - learning_rate * inner
print(w.transpose().tolist()[0])
"""