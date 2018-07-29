from skimage import io
from sklearn.cluster import KMeans
import numpy as np


image = io.imread('touxiang1.jpg')
# io.imshow(image)
# io.show()

# print(image.shape)
# print(image)

rows = image.shape[0]
cols = image.shape[1]

# print('*****************')

image = image.reshape(image.shape[0]*image.shape[1], 3)
# print(image.shape)
# print(image)

# 原来像素点可取值为0-255,现在压缩成128种。
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)

print(clusters.shape)
print(labels[0:5])


labels = labels.reshape(rows, cols)
# np.save('codetouxiang.npy', clusters)
# io.imsave('compressed_test.png', labels)
print(labels[0:5])
