import sys
import csv
import math
import numpy
import pandas

features = ['age', 'sex', 'hours_per_week', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
w = [0.02824522971105768, 0.8559480165328853, 0.030240804570350572, 0.11245194074849818, -0.5238761150309794, -0.03902511651558579, -0.33984016692381663, -0.10614504459195286, -0.8022844858864132, -0.7213586001620965, -0.32962271327077014, -0.5546864736610161, -1.146237058131893, -1.0461844261400501, -0.6660379507392183, -1.3072058257233103, -1.4186319763466337, -1.686114315335822, -1.4584254208622092, 0.23026577140561336, 0.2568477671341428, 0.8627660211829579, 1.9131857198377171, -0.3064455863485799, 1.2412757576686533, -0.667462818069932, 1.850178757038706, 0.043838808135540315, -0.795666641738179, 0.42196556279972836, 0.740309389513188, -0.8410889384385488, -1.2576023204892286, -0.9089279102257454, -0.6633759167152964, -0.13571884096424583, -0.09502051985487978, -0.06811341874360652, 0.6692083478048243, -1.1126247677688474, -0.8639165625937709, -0.46972220369794965, -1.0783123118209048, -0.71579768929138, 0.39540940106060257, 0.39441220261343773, 0.15862515851366676, 0.47985838708947204, -0.26896236746409463, -0.5937115901766991, -0.44317685031018955, -0.41446858003337916, -1.1345943380929395, -1.6002700317363578, -0.6345824261559231, 0.922705451034641, -0.869883347264367, -0.5588094667360657, -0.4983661115259632, -1.0007658294858364, -0.37656202028207036, -3.3043867752941796]
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