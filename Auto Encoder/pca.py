import sys, numpy
from skimage import io, transform

img = numpy.zeros((600*600*3,415))
dir = sys.argv[1]
for i in range (415):
	file_name = dir + '/' + str(i) + ".jpg"
	temp = io.imread(file_name)
	temp = transform.resize(temp, (600, 600, 3))
	temp = temp.flatten()
	img[:,i]=temp

img_mean = numpy.mean(img, axis=1)
for i in range (415):
	img[:,i] = img[:,i] - img_mean

U, s, V = numpy.linalg.svd(img, full_matrices=False)
eigenface = U[:,0:4]
inputimg = sys.argv[2]
choose_img = int(inputimg.replace(".jpg", ""))
weight = numpy.dot(numpy.transpose(img[:,choose_img]), eigenface)
output = numpy.zeros((600 * 600 * 3,))
for i in range(4):
	output = output + eigenface[:,i] * weight[i]

output = output + img_mean - numpy.min(output + img_mean)
output /= numpy.max(output)
output = (output * 255).astype(numpy.uint8)
output = output.reshape(600, 600, 3)
io.imsave("reconstruction.jpg", output)
