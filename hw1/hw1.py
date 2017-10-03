import csv
import sys
import codecs
import pandas

w = [-0.016325495048894263, -0.02870183148689336, 0.21896919137679158, -0.2325277127606827, -0.05766452480800722, 0.5376408752242116, -0.5709474904959306, -0.001878858022979187, 1.1118700277274205, 0.4654965052872306]
index = 0
# read process and get all PM2.5 data
with open(sys.argv[2], 'w', encoding='utf-8') as f:
	spamwriter = csv.writer(f, delimiter=',')
	spamwriter.writerow(['id', 'value'])
	data = pandas.read_csv(sys.argv[1]).values.tolist()
	# start testing x_vector to get y
	for i in range(8, len(data), 18):
		# add bias
		inner_product = w[9]
		# do inner product
		for j in range(0,9):
			inner_product += w[j] * float(data[i][j + 2])
		spamwriter.writerow(['id_'+str(index), inner_product])
		index += 1