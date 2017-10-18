import sys
import csv
import math
import numpy
import pandas

features = ['age', 'sex', 'hours_per_week', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
w = [0.018172074468535936, 0.3995235682887812, 0.0233234308587795, 0.19169441502485926, -0.4210576276299226, -0.005436692360600547, -0.3374169370704488, 0.08895952927393222, -0.6893432157526456, -0.4703316664926803, -0.03204418841904892, -0.5133023594105831, -0.7617836202492582, -0.7716953074987328, -0.31696015350235485, -0.30350363213160136, -0.5027504068443567, -0.8961805013648237, -0.6366518165738985, -0.07984696196300799, -0.05975935513360502, 0.5016486311627957, 0.7957469579682992, -0.5473240164723697, 0.7928347109294598, -0.09621390446945928, 0.9375222588212156, -0.2433616255154354, -0.5904205447142574, 0.04953440406664937, 0.6078552440400605, -0.2555145871240904, -1.203662332159091, -0.45145606499183366, -0.34461486195453966, -0.15205692856428657, -0.011529810709077106, -0.11907075671799165, 0.751507160342176, -0.8049756995831268, -0.6991575393543211, -0.5590313448541206, -1.0518065349791808, -0.13183889053384332, 0.5905281272608726, 0.2139916106377326, 0.21441418347989652, 0.39204686869722766, -0.30256013618790484, -0.5187390517711871, 0.030580652170084214, -0.3722530634379848, -0.5317170337358417, -1.253201877175671, -0.7986673007573223, 0.7369798800996499, -0.32320115263293536, -0.44789975365129236, -0.6230229987371252, -0.33192803424119716, -0.4622268035745713, -2.188278742837146]
x_item = []
y_item = []
# sort the data order
for i in features:
	x_item.append(pandas.read_csv("X_test.csv", sep=',' , usecols=[i]).values.tolist())

# append data into x_item and y_item
for i in range(0, len(x_item[0])):
	tmp = []
	for j in range(0, len(x_item)):
		tmp += x_item[j][i]
	tmp += [1]
	y_item.append(tmp)

x_item = numpy.array(y_item)
y_item = numpy.dot(x_item, w)
y_item = 1 / (numpy.exp(-1 * y_item)+1)

index = 1
with open('output.csv', 'w', encoding='utf-8') as f:
	spamwriter = csv.writer(f, delimiter=',')
	spamwriter.writerow(['id', 'label'])
	# start testing x_vector to get y
	for i in y_item:
		if i < 0.5:
			spamwriter.writerow([str(index), 0])
		else:
			spamwriter.writerow([str(index), 1])
		index += 1