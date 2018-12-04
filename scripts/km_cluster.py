import matplotlib.pyplot as plt

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
import numpy as np


def centroid_dist(sample):
    return np.array([np.linalg.norm(sample - c) for c in k_means.cluster_centers_])

best_samp = None
min = np.inf
c_d_best = None
# random_sample = tX_test[0].copy()
random_sample = np.random.uniform(low=-3, high=1, size=(196,))

best_pred = predict_model.predict(random_sample.reshape(1, 14, 14, 1), batch_size=1)

fig, ax = plt.subplots(nrows=2)

ax[0].imshow(best_pred.reshape(28, 28))
ax[1].imshow(random_sample.reshape((14,14)))
# %%
from IPython.display import display, clear_output

print("before: {:.4f}".format(np.linalg.norm(random_sample - k_means.cluster_centers_[0])))

fig, ax = plt.subplots()
plot_ind = np.array([j*1e5 for j in range(10)])

for i in range(500000):

    n_change = np.random.randint(low=0, high=195)
    change_int = np.random.randint(low=0, high=195, size=(n_change,))
    changes = np.random.uniform(low=-1, high=1, size=(n_change,))
    old = random_sample.copy()
    random_sample[change_int] += changes

    c_d = np.linalg.norm(random_sample - k_means.cluster_centers_[4])
    c_d += 0.01*np.sum(random_sample)

    c_o = np.linalg.norm(old - k_means.cluster_centers_[4])
    c_o += 0.01*np.sum(old)
    eps = np.random.uniform()

    if (c_d/c_o) > eps:
        random_sample = old

    if any(i== j for j in plot_ind):
        best_pred = predict_model.predict(random_sample.reshape(1, 14,14,1), batch_size=1)
        ax.imshow(best_pred.reshape(28, 28))
        display(fig)

plt.show()
print("after: {:.4f}".format(np.linalg.norm(random_sample - k_means.cluster_centers_[0])))
# %%


best_pred = predict_model.predict(random_sample.reshape(1, 14,14,1), batch_size=1)

fig, ax = plt.subplots(nrows=2)

ax[0].imshow(best_pred.reshape(28, 28))
ax[1].imshow(random_sample.reshape((14,14)))

# %%

fig, axs = plt.subplots(5, 2, figsize=(12, 12))
cent_img = cent_img.reshape((5, 2, 28, 28))

for i in range(5):
    for j in range(2):
        axs[i][j].imshow(cent_img[i][j])
