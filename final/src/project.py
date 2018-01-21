import sys, re, jieba, numpy, gensim, csv, logging
from itertools import islice


# load stopwords
stopwordset = set()
with open('stopwords.txt', 'r') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

		
# training
if sys.argv[1] != sys.argv[1]:
	sentences = []
	print("start to process training data")
	with open('training data/word_vector','r') as f:
		for line in f:
			line = re.sub("\n", "", line)
			line = re.split(" ", line)

			# remove stopwords in the sentence
			line2 = []
			for word in line:
				if word not in stopwordset:
					line2.append(word)
			sentences.append(line2)

	print("start to word2vec training")
	logging.basicConfig(format='%(message)s', level=logging.INFO)
	model = gensim.models.word2vec.Word2Vec(sentences, size = 300, window = 30, min_count = 12, hs = 1, iter = 30, sample = 1e-4)
	model.save("word_vector_model")


# testing
if sys.argv[1] == sys.argv[1]:
	print("start to test data")
	# load jieba dictionary and model
	jieba.set_dictionary('dict.txt.big')
	model = gensim.models.Word2Vec.load("word_vector_model")
	
	with open(sys.argv[2], 'w', encoding='utf-8') as wf:
		spamwriter = csv.writer(wf, delimiter=',')
		spamwriter.writerow(['id', 'ans'])
		with open(sys.argv[1], 'r') as rf:
			for iter, line in enumerate(islice(rf, 1, None)):  
				line = re.split("^[\d]+,", line.strip())[1]
				line = re.sub("[A-Z]:","",line)
				line = re.split(",", line)

				# get the test sentence vector
				ans_score = numpy.zeros((1, 300))
				for i in jieba.cut(line[0]):
					if i not in stopwordset:
						try:
							ans_score += model[i]   
						except:
							pass

				max_similarity = -2
				max_index = 0
				line = re.split("\t", line[1])
				for i, sentences in enumerate(line):
					# get the option sentence vector
					score = numpy.zeros((1, 300))
					for j in jieba.cut(sentences):
						if j not in stopwordset:
							try:
								score += model[j]
							except:
								pass

					# find the cosθ between test sentence and option sentence and find the more similarity
					score = numpy.transpose(score)
					similarity = (numpy.dot(ans_score, score)[0][0] / numpy.linalg.norm(ans_score) / numpy.linalg.norm(score))
					if similarity > max_similarity:
						max_similarity = similarity
						max_index = i
				
				# write the prediction and show progress bar
				iter += 1
				spamwriter.writerow([str(iter), str(max_index)])
				sys.stdout.write('  \033[1;34m╠' + '█' * int((iter / 5060) * 50) + '░' * (50 - int((iter / 5060 * 50))) + "╣\033[0m " + str('{:3d}'.format(int((iter / 5060) * 100))) + "%\r")
				sys.stdout.flush()
	print()