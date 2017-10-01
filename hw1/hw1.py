import csv
import sys
import codecs
import pandas

training_time = 10
learning_rate = 0.2
w = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x_item = list()
y_item = list()

# read process and split the data into x_vector and y_vector(PM2.5)
for i in range(3,27):
	col = pandas.read_csv(r"train.csv",usecols=[i]).values.tolist()
	for i in range(0, len(col), 18):
		if 'NR' == ''.join(col[i+10]):
			col[i + 10].clear()
			col[i + 10].append('0')
		x_item.append(col[i : i + 9] + col[i + 10: i + 18] + [['1']])
		y_item.append(col[i + 9: i + 10][0][0])
		
# start training w
data_len = len(y_item)
for i in range(0,training_time):
	for j in range(0,data_len):
		# get wx
		inner_product = 0
		total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		for k in range(0,18):
			inner_product = inner_product + w[k] * float(x_item[j][k][0])
		# get y - wx
		inner_product = float(y_item[j]) - inner_product
		
		# get Σ(y - wx)x 
		for k in range(0,17):
			total[k] = total[k] + learning_rate * inner_product * float(x_item[j][k][0])
			
		# Constant term
		total[17] = total[17] + learning_rate * inner_product
		
	#renew here w
	for k in range(0,18):
		w[k] = w[k] + total[k] / data_len

for k in range(0,18):	
	print(w[k])