from keras.datasets import mnist
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape
from keras.layers import ZeroPadding2D, ZeroPadding3D
import keras.regularizers as reg
import keras.optimizers as opt

import matplotlib.pyplot as plt
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)/255
X_test = X_test.reshape(10000, 28, 28, 1)/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# %%
model = Sequential()
model.add(Conv2D(
            filters=3,
            kernel_size=2,
            strides=1,
            padding="same",
            activation="relu",
            use_bias=True,
            kernel_initializer="random_uniform",
            input_shape=(28, 28, 1),
            kernel_regularizer=reg.l2(0.01),
            bias_regularizer=reg.l2(0.01)
            ))

model.add(Conv2D(
            filters=1,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            use_bias=True,
            kernel_initializer="random_uniform",
            input_shape=(28, 28, 3),
            kernel_regularizer=reg.l2(0.01),
            bias_regularizer=reg.l2(0.01)
            ))

model.add(MaxPool2D(
        pool_size=(2, 2)
        )
        )

model.add(Flatten())

model.add(Dense(
        196,
        activation="relu",
        use_bias=True,
        kernel_regularizer=reg.l2(0.01),
        bias_regularizer=reg.l2(0.01)
    )
    )

model.add(Reshape((14, 14, 1)))

model.add(ZeroPadding2D(4))
model.add(Conv2D(
        filters=5,
        kernel_size=3,
        strides=1,
        use_bias=True,
        activation="relu",
        kernel_regularizer=reg.l2(0.01),
        bias_regularizer=reg.l2(0.01)
        ))

model.add(ZeroPadding2D(5))
model.add(Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        kernel_regularizer=reg.l2(0.01),
))

model.compile(
        optimizer=opt.adam(lr=0.01, amsgrad=False),
        loss="binary_crossentropy",
        )


 # %%

model.fit(
            X_train,
            X_train,
            validation_data=(X_test, X_test),
            epochs=3,
            batch_size=100
    )

# %%
images = X_test[:2]
print(images.shape)

pred_images = model.predict(images)
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
