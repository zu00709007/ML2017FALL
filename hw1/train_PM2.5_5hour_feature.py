import csv
import sys
import codecs
import pandas

training_time = 100000
learning_rate = 0.00001
w = [0] * 6
x_item = list()
y_item = list()
index = 0

# read process and get all PM2.5 data
data = pandas.read_csv(r"train.csv").values.tolist()
for i in range(9, len(data), 18):
	x_item += data[i][3:28]
	index += 1
	if index == 20:
		y_item.append(x_item)
		x_item = list()
		index = 0

data = y_item
x_item = list()
y_item = list()
# read process and split the data into x_vector(first 5 hours PM2.5) and y_vector(10th hour PM2.5) and add constant
for i in range(0, len(data)):
	for j in range(0, len(data[i]) - 5):
		x_item.append(data[i][j : j + 5] + [1])
		y_item.append(data[i][j + 5])

# start training w
data = len(y_item)
for i in range(0,training_time):
	total = [0] * 6
	cost= 0
	for j in range(0,data):
		# get wx by using inner product
		inner_product = 0
		for k in range(0,6):
			inner_product += w[k] * float(x_item[j][k])		
		# get y - wx
		inner_product -= float(y_item[j])		
		# get total cost
		cost = cost + inner_product ** 2		
		# get Σ(y - wx)x 
		for k in range(0,5):
			total[k] += learning_rate * inner_product * float(x_item[j][k])			
		# constant
		total[5] += learning_rate * inner_product			
	# update w
	for k in range(0,6):
		w[k] = w[k] - total[k] / data	
		
print(cost/(2*data))
print(w)