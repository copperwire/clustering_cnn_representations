import keras as K
from keras.datasets import mnist
from keras.utils import to_categorical

m_loc = "/home/solli-comphys/github/clustering_cnn_representations/models/mnist_cnn.h5"
model = K.models.load_model(m_loc)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)/255
X_test = X_test.reshape(10000, 28, 28, 1)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%

for i in range(5):
    model.pop()
print(model.output_shape)

# %%

tX_train = model.predict(X_train)
tX_test = model.predict(X_test)

# %%

from sklearn.cluster import KMeans

k_means = KMeans(
            n_clusters=10,
            random_state=42,
            n_jobs=4,
            precompute_distances=True
            )

k_means.fit(tX_train)

# %%

m_loc = "/home/solli-comphys/github/clustering_cnn_representations/models/mnist_cnn.h5"
reconstruct_model = K.models.load_model(m_loc)
print(reconstruct_model.summary())

# %%

predict_model = K.models.Sequential()
for i in reconstruct_model.layers[6:]:
    predict_model.add(i)

predict_model.build(input_shape=(None, 14, 14, 1))
print(predict_model.summary())
# %%
from sklearn.manifold import TSNE

t_sne = TSNE(n_components=2)
t_sne.fit(tX_test)
pr_test = t_sne.transform(tX_test)
pr_clusters = t_sne.transform(k_means.cluster_centers_)

# %%

centroids = k_means.cluster_centers_.reshape(10, 14, 14, 1)
cent_img = predict_model.predict(centroids)

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5, 2, figsize=(12, 12))
cent_img = cent_img.reshape((5, 2, 28, 28))

for i in range(5):
    for j in range(2):
        axs[i][j].imshow(cent_img[i][j])
