from keras.datasets import mnist
from keras.utils import to_categorical

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape
from keras.layers import Conv2DTranspose
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.layers import Input, Lambda

from keras.losses import mse, binary_crossentropy

import keras.regularizers as reg
import keras.optimizers as opt
from keras.utils import plot_model

from keras import backend as K
import keras as ker

import matplotlib.pyplot as plt
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)/255
X_test = X_test.reshape(10000, 28, 28, 1)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# %%


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# %%
kernel_size = 4
filters = 20
latent_dim = 10
num_layers = 2

in_layer = Input(shape=(28, 28, 1))
h1 = in_layer
shape = K.int_shape(h1)

for i in range(1, num_layers+1):
    filters *= 2
    h1 = Conv2D(
            filters,
            kernel_size,
            activation="relu",
            strides=2,
            padding="same"
            )(h1)

shape = K.int_shape(h1)
h1 = Flatten()(h1)

h1 = Dense(16, activation='relu')(h1)
mean = Dense(latent_dim)(h1)
var = Dense(latent_dim)(h1)

sample = Lambda(sampling, output_shape=(latent_dim,))([mean, var])

encoder = Model(in_layer, [mean, var, sample], name="encoder")
encoder.summary()

# %%
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
de1 = Dense(
        shape[1] * shape[2] * shape[3],
        activation='relu')(latent_inputs)

de1 = Reshape((shape[1], shape[2], shape[3]))(de1)

for i in reversed(range(1, num_layers+1)):
    de1 = Conv2DTranspose(
                        filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same'
                        )(de1)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(de1)
decoder = Model(input=latent_inputs, output=outputs)
decoder.summary()
# %%

outputs = decoder(encoder(in_layer)[2])
vae = Model(in_layer, outputs, name='vae')


def vae_loss(y_true, y_pred):
    xent_loss = 784 * binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


vae.compile(optimizer="adam", loss=vae_loss)

repo = "/home/solli-comphys/github/clustering_cnn_representations/"
plot_model(vae, show_shapes=True, to_file=repo+"images/vae.png")

# %%

earlystop = ker.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=2,
                            patience=0,
                            verbose=0,
                            mode='auto',
                            restore_best_weights=True
                              )

vae.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=20,
        batch_size=1000,
        callbacks=[earlystop,]
    )

# %%
num_img = 12
images = X_test[num_img:num_img+3]
print(images.shape)

pred_images = vae.predict(images)
image_0 = images[0].reshape((28, 28))
image_1 = images[1].reshape((28, 28))

pred_image_0 = pred_images[0].reshape((28, 28))
pred_image_1 = pred_images[1].reshape((28, 28))

f, ax = plt.subplots(nrows=2, ncols=2)

ax[0][0].imshow(image_0)
ax[0][1].imshow(image_1)

ax[1][0].imshow(pred_image_0)
ax[1][1].imshow(pred_image_1)

plt.show()

# %%

vae.save("/home/solli-comphys/github/clustering_cnn_representations/models/mnist_vae.h5")
encoder.save("/home/solli-comphys/github/clustering_cnn_representations/models/mnist_enc.h5")
decoder.save("/home/solli-comphys/github/clustering_cnn_representations/models/mnist_dec.h5")

# %%
import numpy as np

num_plots = 10

fig, ax = plt.subplots(nrows = num_plots, ncols = num_plots, figsize=(12, 12))

# %%
ms = np.random.normal(0, 2, size=(num_plots**2, latent_dim))

for i in range(num_plots):
    for j in range(num_plots):
        inp = ms[(i + 1) * j]
        inp = inp.reshape((1, latent_dim))
        out = decoder.predict(inp)
        ax[i][j].imshow(out.reshape((28, 28)))
        ax[i][j].set_xticks([])
        ax[i][j].set_yticks([])

fig.savefig(repo+"/images/generated_digits.png")
