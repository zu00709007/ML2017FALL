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
training_time = 11111111
learning_rate = 1e-7
#w = [0] * len(x_item[0])
w = [0.02824522971105768, 0.8559480165328853, 0.030240804570350572, 0.11245194074849818, -0.5238761150309794, -0.03902511651558579, -0.33984016692381663, -0.10614504459195286, -0.8022844858864132, -0.7213586001620965, -0.32962271327077014, -0.5546864736610161, -1.146237058131893, -1.0461844261400501, -0.6660379507392183, -1.3072058257233103, -1.4186319763466337, -1.686114315335822, -1.4584254208622092, 0.23026577140561336, 0.2568477671341428, 0.8627660211829579, 1.9131857198377171, -0.3064455863485799, 1.2412757576686533, -0.667462818069932, 1.850178757038706, 0.043838808135540315, -0.795666641738179, 0.42196556279972836, 0.740309389513188, -0.8410889384385488, -1.2576023204892286, -0.9089279102257454, -0.6633759167152964, -0.13571884096424583, -0.09502051985487978, -0.06811341874360652, 0.6692083478048243, -1.1126247677688474, -0.8639165625937709, -0.46972220369794965, -1.0783123118209048, -0.71579768929138, 0.39540940106060257, 0.39441220261343773, 0.15862515851366676, 0.47985838708947204, -0.26896236746409463, -0.5937115901766991, -0.44317685031018955, -0.41446858003337916, -1.1345943380929395, -1.6002700317363578, -0.6345824261559231, 0.922705451034641, -0.869883347264367, -0.5588094667360657, -0.4983661115259632, -1.0007658294858364, -0.37656202028207036, -3.3043867752941796]
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
