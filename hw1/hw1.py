import csv
import sys
import codecs
import pandas

training_time = 1000
learning_rate = 0.0001
w = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x_item = list()
y_item = list()

# read process and split the data into x_vector and y_vector(PM2.5) and add constant
for i in range(3,27):
	col = pandas.read_csv(r"train.csv",usecols=[i]).values.tolist()
	for j in range(0, len(col), 18):
		if 'NR' == ''.join(col[j+10]):
			col[j + 10].clear()
			col[j + 10].append('0')
		x_item.append(col[j : j + 9] + col[j + 10: j + 18] + [['1']])
		y_item.append(col[j + 9: j + 10][0][0])
		
# start training w
data_len = len(y_item)
for i in range(0,training_time):
	total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	cost= 0
	for j in range(0,data_len):
		# get wx by using inner product
		inner_product = 0
		for k in range(0,18):
			inner_product = inner_product + w[k] * float(x_item[j][k][0])
		
		# get y - wx
		inner_product = inner_product - float(y_item[j])
		
		# get total cost
		cost = cost + inner_product * inner_product
		
		# get Σ(y - wx)x 
		for k in range(0,17):
			total[k] = total[k] + learning_rate * inner_product * float(x_item[j][k][0])
			
		# constant
		total[17] = total[17] + learning_rate * inner_product
		
	
	# update w
	for k in range(0,18):
		w[k] = w[k] - total[k] / data_len
		
print(cost/(2*data_len))
#print(w)	
