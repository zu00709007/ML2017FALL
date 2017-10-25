import sys
import csv
import math
import numpy
import pandas

# below is testing code
features = ['age', 'fnlwgt', 'sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
w = [14.480764465493362, 1.3052615136893677, 0.8741046818022903, 32.039718484850134, 2.846224983689413, 0.1604649770040801, -0.5063786699292041, -0.601787297871097, -0.2665480638971222, 0.041468355969073854, -0.7012862271206965, -0.6687277446062315, -3.0174738769622564, -0.13759579131268931, -0.7680189832722496, -0.7275869308706971, -0.3547954699752086, -1.59293603225184, -1.265419482106108, -1.2334031174505038, -1.0355770769270776, 0.45750382268729584, 0.467116811230327, 1.0692320423145087, 2.1804509203552693, -0.03100748829498291, 1.3589181395154593, -6.549047306481089, 2.007077142234616, 0.319628670566209, -0.10964785953016594, -1.0550053063648823, -0.05412834014172488, 0.7717035103037725, -0.8146883908391986, -0.7911124299610269, -0.4296483647352409, -0.9578926756995233, -3.2519950626371146, 0.4433417621379267, 0.6291820289707682, 0.27838306974739924, 0.5135841334741569, -0.13055732426746997, -0.73938308918379, 0.2470479746732755, -1.5347021204955522, -1.6773078967970192, -2.509717282761468, -1.6832969894861989, 1.4601119761407582, -1.367652020259864, -0.9575290688064306, -1.0686699947958525, -1.5108468343848975, -0.793166420479142, -3.838174293365524, -1.2612535187056952, -10.525401722072887, -5.6978643387262675]
w = [15.934917582125511, 1.4081119841209881, 0.8755359674051237, 32.475697351869435, 2.84916117971882, 0.2281635681985599, -0.4408718725267887, -0.8920373634206769, -0.20029516847374618, 0.10679117207843328, -0.636754589175673, -0.6022819850159132, -3.6758707947194122, 0.05278888908608706, -0.6021804017197666, -0.5640643713736095, -0.18889515608970286, -1.4229956762684441, -1.0997684898221538, -1.0676865542657659, -0.8692240115334101, 0.6208465014354584, 0.630228719383042, 1.2331547171831598, 2.345803136143706, 0.13291793074292196, 1.523016178624274, -9.386357862475744, 2.170897877519711, 0.4839393185472818, -0.08616736617277507, -1.1719322698892733, -0.030576709187627296, 0.7957866854171967, -0.7829372584338853, -0.7678664704683746, -0.40687541373710123, -0.9336434154630938, -3.6913976884685327, 0.46682816662199345, 0.6536354210174239, 0.3029892725651639, 0.5375552491289959, -0.10651787256454869, -0.8392484743345918, 0.1818444397467347, -1.6002842479509647, -1.7203741690940828, -2.5702261866472282, -1.7490236702207917, 1.3976956901971234, -1.4404948652607192, -1.0293977275299782, -1.1407698959805268, -1.5855812586095053, -0.8641243965884222, -6.676297897161008, -1.5440749646770928, -8.774081036112685, -6.060368143969268]
features = ['age', 'sex', 'capital_gain', 'capital_loss', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia', '?_native_country']
w = [21.73778862881497, 0.9409838650535821, 31.740171234231127, 2.857823579577391, 0.13627368342968207, -0.5264627452962721, -0.6327495821001561, -0.281256726459657, 0.020939648721583114, -0.717846300285447, -0.6856419733722149, -3.043401589102654, -0.13071853673960884, -0.8180081647391907, -0.7748641106250074, -0.41468847771664663, -1.4323048556583793, -1.0605763203251577, -1.219407094568774, -1.0370868842226055, 0.3692552762665414, 0.3759427848736045, 0.9987566365534195, 2.1716784714322674, -0.10659026139487993, 1.3126393456754923, -6.42565023171333, 1.9579596712675789, 0.24208009369033642, -1.6458752129533014, 1.5801793920866005, 0.629315184143917, -1.6018603240755191, -1.8780515362764096, -1.7654064362839275, -1.1791651878461165, -0.10017057271551229, -1.0809464662900259, -0.05655276262183404, 0.769266830199618, -0.8397028614895206, -0.7738066255250629, -0.4137698105057858, -0.9388582169903604, -3.409070476419842, 0.45024892267124283, 0.6204788505344914, 0.280832356743955, 0.5275258101214417, -0.13287098007778914, -0.7634681188397657, -1.2049438517567972, -0.6070453540857721, -1.5360426166233598, -1.7366309940868243, -0.8362246027532353, 0.06002329810123094, -1.5587050339138773, -0.873526049969678, -1.1301588236576898, -1.3715399229159944, -0.9269342907474556, 1.230285593909671, 0.43777937236615977, -0.6973699472235751, -1.833159711259023, 0.44303883466847915, -1.6881290227416148, 0.10965171350395328, -0.6114397477700019, 0.5129247836991083, 0.7305055619049364, 0.5391729888969109, -0.9056629706514376, -0.16800630172847758, -0.11455093973558703, -0.08272937270768876, -0.7962423345570877, 0.11912343053718918, 0.09025124193496777, -0.3351062315246243, 0.10094215950461621, 0.8075646662992307, 0.8663396958467453, 0.08078345038032231, 0.42646333470629527, -0.3759072831752408, -0.4743937433905495, -0.6374073383208001, -1.914736277502842, -0.8837793025657784, 0.48583179219684075, 0.13217122711295645, 0.06462468689972045, -0.39581827314058865, 0.08688321655898007, -0.8714907914260325, 0.01520316174407533, 0.012036592729223418, -0.39694632038449745, 0.28171044160793296, -1.0107725993873715, 0.8902596078404786, -0.1307631668606633, -19.625574766028166, -5.86086412120477]
x_item = []
y_item = []
# sort the data order
for i in features:
	x_item.append(pandas.read_csv(sys.argv[5], sep=',' , usecols=[i]).values.tolist())

# data normalization
tmp = numpy.array(x_item[0])
tmp = tmp / 90
x_item[0] = tmp.tolist()
"""tmp = numpy.array(x_item[1])
tmp = tmp / 1484705
x_item[1] = tmp.tolist()"""
tmp = numpy.array(x_item[2])
tmp = tmp / 99999
x_item[2] = tmp.tolist()
tmp = numpy.array(x_item[3])
tmp = tmp / 4356
x_item[3] = tmp.tolist()

# square item
row_name = [0]
for i in row_name:
	tmp = numpy.array(x_item[i])
	tmp = tmp ** 2
	x_item.append(tmp.tolist())

# 3rd power
"""row_name = [0]
for i in row_name:
	tmp = numpy.array(x_item[i])
	tmp = tmp ** 3
	x_item.append(tmp.tolist())"""

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