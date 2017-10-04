import csv
import sys
import codecs
import pandas

training_time = 10000
learning_rate = 0.000003
w = [0] * 163
x_item = [[]] * 18
y_item = list()
index = 0

# read process and get all data
data = pandas.read_csv(r"train.csv").values.tolist()
for i in range(0, len(data)):
	x_item[i%18] = x_item[i%18] + data[i][3:28]
	index += 1
	if index == 360:
		y_item.append(x_item)
		x_item = [[]] * 18
		index = 0

data = y_item
x_item = list()
y_item = list()
index = list()
# read process and split the data into x_vector(first 9 hours all) and y_vector(10th hour PM2.5) and add constant
for i in range(0, len(data)):
	for j in range(0, len(data[0][0]) - 9):
		for k in range(0, 18):
			index = index + data[i][k][j : j + 9]
		index = index + [1]
		x_item.append(index)
		y_item.append(data[i][9][j + 9])
		index = list()

# start training w
data = len(y_item)
for i in range(0,training_time):
	total = [0] * 163
	cost= 0
	for j in range(0,data):
		# get wx by using inner product
		inner_product = 0
		for k in range(0,163):
			inner_product += w[k] * float(x_item[j][k])		
		# get y - wx
		inner_product -= float(y_item[j])		
		# get total cost
		cost = cost + inner_product ** 2		
		# get Σ(y - wx)x 
		for k in range(0,162):
			total[k] += learning_rate * inner_product * float(x_item[j][k])			
		# constant
		total[162] += learning_rate * inner_product		
	# update w
	for k in range(0,163):
		w[k] = w[k] - total[k] / data	
		
print(cost/(2*data))	
print(w)