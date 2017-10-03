import csv
import sys
import codecs
import pandas

w = [1] * 18
x_item = list()
y_item = [0] * 240
# read process and split the data into x_vector
for i in range(8,11):
	col = pandas.read_csv(r"test.csv", header=-1, usecols=[i]).values.tolist()
	for j in range(0, len(col), 18):
		if 'NR' == ''.join(col[j+10]):
			col[j + 10].clear()
			col[j + 10].append('0')
		x_item.append(col[j : j + 9] + col[j + 10: j + 18])
		
# start testing x_vector to get y_vector
data_len = len(x_item)
for i in range(0,data_len):
	# add bias
	inner_product = w[17]
	for j in range(0,17):
		inner_product += w[j] * float(x_item[i][j][0])
	y_item[i%240] += inner_product
	

with open('output.csv', 'w', encoding='utf-8') as f:
	spamwriter = csv.writer(f, delimiter=',')
	spamwriter.writerow(['id', 'value'])
	for i in range(0, 240):
		spamwriter.writerow(['id_'+str(i), y_item[i]/3])