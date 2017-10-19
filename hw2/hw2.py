import sys
import csv
import math
import numpy
import pandas


# use [feature] to select what you want from train data
# features below use this type
# 'age':0, 'fnlwgt':0, 'education_num':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0
dic = {'age':0, 'fnlwgt':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0}
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
row_name = [0, 1, 3, 4, 5]
for i in row_name:
	tmp = numpy.array(y_item[i])
	print(numpy.amax(tmp))
	tmp = tmp / numpy.amax(tmp)
	y_item[i] = tmp.tolist()
	
# square item
row_name = [0, 1]
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
w = [13.294908910361189, 1.679117400245717, 0.891738088700712, 31.742966069548814, 2.841457549845557, 2.4245578495963924, 0.13325804728123009, -0.5303597659522019, -0.6626571763462424, -0.29253491732842973, -0.10675834515780147, -0.7699701692377924, -0.6455001591157719, -2.8716445931759624, -0.037102390616819914, -0.8367383030560973, -0.7365403598216771, -0.4062522368271007, -1.3856260957806261, -1.0743389533730727, -1.255126799621908, -1.0230519196762506, 0.3994857663261761, 0.41933199632575696, 1.0292799170565525, 2.1073774574841844, -0.08439610618918608, 1.3256989984423315, -6.454724910782033, 1.9338275041380515, 0.25852457570502546, -1.6431807159269902, 1.512980161931228, 0.6523329690722679, -1.5919138845388463, -1.8612946629131526, -1.736984540961312, -1.1152087963130153, -0.03913876123737527, -1.075217399444325, -0.014934974274664168, 0.749494807971544, -0.9937055429427616, -0.7024458047700297, -0.36546414225303436, -0.8583062501958351, -3.5528832701940467, 0.47495482243301473, 0.603218528820364, 0.2661039876887204, 0.6002053709173991, -0.17539127520577166, -0.6997595669630455, -1.233410641825384, -0.6453746726467899, -1.505157280435524, -1.7027412573908054, -0.8426637666357762, 0.14607814928443866, -1.5147385795490373, -0.8442297188395056, -1.158064798791046, -1.3326487608813045, -0.9335876115889095, 1.290868047583727, 0.45221624710341096, -0.6397444867735957, -1.9185414453192162, 0.47845674771292007, -1.7552502475641854, 0.20786963898256633, -0.5699070923220392, 0.5232145978927191, 0.6828514746579045, 0.5508367299830681, -0.9244294480626501, -0.124709247781231, -0.0334510947880968, -0.07310815996608651, -0.7864592530027908, 0.1099142658004564, 0.12719593610187951, -0.2944637038948738, 0.10386566588588285, 0.8407937254821588, 0.8384533160039704, 0.14574789641030153, 0.38380130401308565, -0.4023941255330591, -0.49758790258603236, -0.6457712402655637, -1.9192148550736126, -0.7538700153549167, 0.4877876816145731, 0.1984770075580218, 0.10392394648676676, -0.33973757980178615, 0.020678528830317525, -0.9631159850397673, 0.12737539829866723, -0.4170686133950657, -0.3904245984084965, 0.30132791243796186, -0.99440439477197, 0.7958935921447301, -0.11116564092982542, -3.3958261945851183, -1.6131753018589974, -9.363933836543811, -5.783269469649896]
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
