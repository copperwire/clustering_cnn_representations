import numpy as np
from keras.utils import Sequence
from scipy.ndimage import imread


class SampleSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.sample_filenames, self.targets = x_set, y_set
        self.batch_size = batch_size
        self.distr = lambda x, lmbd: -(1/lmbd)*np.log(x*(1/lmbd))

        self.corruption_rate = 0.3
        self.dim = 20*20*20
        self.n_noise =int(np.ceil(self.corruption_rate*self.dim))

    def __len__(self):
        return int(np.ceil(len(self.sample_filenames)/self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.sample_filenames[
            idx * self.batch_size:
            (idx + 1) * self.batch_size]
        """
        batch_y = self.targets[
            idx * self.batch_size:
            (idx + 1) * self.batch_size]
        """
        X = np.array([imread(file_name) for file_name in batch_x])
        #dim = (self.batch_size, self.n_noise)
        #base = np.zeroes(dim)

        #input = np.random.uniform(1e-6, 1, size=X.shape)
        # X_noised = X + self.distr(input, 1.5)
        return X, X # np.array(batch_y)
