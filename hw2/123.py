import sys
import csv
import math
import numpy
import pandas

features = ['age', 'sex', 'hours_per_week', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White']
features = ['age', 'sex', 'hours_per_week', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia', '?_native_country']
w = [0.018172074468535936, 0.3995235682887812, 0.0233234308587795, 0.19169441502485926, -0.4210576276299226, -0.005436692360600547, -0.3374169370704488, 0.08895952927393222, -0.6893432157526456, -0.4703316664926803, -0.03204418841904892, -0.5133023594105831, -0.7617836202492582, -0.7716953074987328, -0.31696015350235485, -0.30350363213160136, -0.5027504068443567, -0.8961805013648237, -0.6366518165738985, -0.07984696196300799, -0.05975935513360502, 0.5016486311627957, 0.7957469579682992, -0.5473240164723697, 0.7928347109294598, -0.09621390446945928, 0.9375222588212156, -0.2433616255154354, -0.5904205447142574, 0.04953440406664937, 0.6078552440400605, -0.2555145871240904, -1.203662332159091, -0.45145606499183366, -0.34461486195453966, -0.15205692856428657, -0.011529810709077106, -0.11907075671799165, 0.751507160342176, -0.8049756995831268, -0.6991575393543211, -0.5590313448541206, -1.0518065349791808, -0.13183889053384332, 0.5905281272608726, 0.2139916106377326, 0.21441418347989652, 0.39204686869722766, -0.30256013618790484, -0.5187390517711871, 0.030580652170084214, -0.3722530634379848, -0.5317170337358417, -1.253201877175671, -0.7986673007573223, 0.7369798800996499, -0.32320115263293536, -0.44789975365129236, -0.6230229987371252, -0.33192803424119716, -0.4622268035745713, -2.188278742837146]
w = [0.015488288537732722, 0.31741979330608905, 0.021381322412353843, 0.19557379149213183, -0.3848233998622576, -0.004569365132309071, -0.32985804348876974, 0.13096354331344628, -0.6498803957151472, -0.4104029593332068, -0.025931460710586694, -0.4886476604136572, -0.6741719117757023, -0.693007048367784, -0.27436262918071475, -0.24440631984428543, -0.4094758753685943, -0.7821785631955255, -0.5470347363160443, -0.08496783715417952, -0.07305187819361635, 0.4701703720135738, 0.6869604591167925, -0.5741971292141762, 0.7509595936751278, -0.07740073137380449, 0.829882749438435, -0.2712944641097791, -0.5508267128298306, 0.04105535230829852, 0.635985802244417, -0.21433213864506073, -1.191584575967397, -0.39390578994486986, -0.2939678870159065, -0.16131775156370848, -0.00912522303713102, -0.12515694845971218, 0.7597380015486128, -0.7176989148560945, -0.6330309509264803, -0.5356144691311366, -0.9954198450311202, -0.1122436783517191, 0.6053461713289224, 0.17396095727647728, 0.2193602161520575, 0.34841771012358347, -0.29157419937697737, -0.4932170255459523, 0.0994086294576844, -0.34221571468036016, -0.4656392083164945, -1.1720581013084665, -0.7671834912534172, 0.6801119362507493, -0.2594257347036681, -0.4201562657478754, -0.5796912036052089, -0.2719202196824307, -0.4363825261111488, 0.014852905524675069, 0.009276578258113666, -0.08785404160854601, -0.08385454942412804, -0.02237623643268904, -0.08732397173043394, -0.02589646704038171, -0.07889397671470236, 0.014002836367225355, 0.015694933679312774, 0.013611497035537105, -0.03458165328051409, -0.0474395526660964, -0.029680044009495213, -0.0004051848388025529, -0.008092915565611234, -0.017106234571753264, -0.0060562337060427315, -0.06598639362094275, -0.0030286556650780698, 0.0016904977419075048, 0.007638856647498176, -0.04002076580465599, 0.010471223713177245, -0.02321031808810969, -0.6041314484017851, -0.03950757726887419, -0.023381286607633455, -0.0276069628479707, -0.005539142278742912, -0.03453036774047125, -0.04485474866840258, -0.1055475662906855, -0.00260571586504971, -0.0861636970851059, -0.019419784514853645, -0.019397426130629933, -0.0181244330692564, -0.053244203927045576, -0.07495235031903005, 0.0023473653872894793, -0.23634873842153895, -1.967575949850347]
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