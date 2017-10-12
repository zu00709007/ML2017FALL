import csv
import sys
import codecs
import pandas

w = [0.23434202847177152, -0.0040185941750191592, -0.006996585136256589, -0.0033706868269028764, 0.011535158143121839, 0.013495087287629107, -0.010973769435844516, -0.014248682992899477, 0.029015288875831498, 0.095618797300287078, 0.02295653111883229, -0.0053146111442841848, 0.015098082779189422, -0.028639198275941885, 0.021922938207050276, 0.072310382515189287, -0.12959557745079464, 0.15463057496575294, 0.65551217639279968, 5.1538500959085595e-05, -1.829304740518369e-05, 5.0362191871962603e-05, 7.4644232185191916e-05, -0.0005257913987939401, 0.00021093461081474207, 0.00026736503572734392, -0.00060784177415521478, 0.00015462530926372604, -0.00080093687256120767, 0.00021702299248305509, 0.0017667491973643995, -0.0020515188486366354, -8.630782500879171e-05, 0.0043796221752636783, -0.0052966766480809476, -0.0013957304705574335, 0.0040952831741559934]
index = 0

with open(sys.argv[2], 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['id', 'value'])
    data = pandas.read_csv(sys.argv[1], header = -1).values.tolist()
    len_data = len(data)
    # start testing x_vector to get y
    for i in range(0, len_data, 18):
        x = []
        x += data[i+8][2:11]
        x += data[i+9][2:11]
        for j in range(i+8, i+10):
            for k in range(2, 11):
                x.append(float(data[j][k]) ** 2)

        # add bias
        inner_product = w[0]
        # do inner product
        for m in range(1, 37):
            inner_product += w[m] * float(x[m-1])
        spamwriter.writerow(['id_'+str(index), inner_product])
        index += 1
