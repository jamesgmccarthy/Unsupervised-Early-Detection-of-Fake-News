"""
Code is adapted from https://github.com/XifengGuo/IDEC/blob/master/DEC.py
Implementation of Deep Embedded Clustering (DEC) algorithm:
    Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.
"""

from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    :param y_true: true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    :return: accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def dec_autoencoder(dims, act='relu'):
    """
    Fully connected auto-encoder model, symmetric
    :param dims: list of number of units in each layer of encoder.  dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
    :param act: activation, not applied to Input, Hidden and Output layers
    :return: Model of autoencoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x
    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name=f'encoder_{i}')(h)

    # hidden layer
    # hidden layer, features are extracted from here
    h = Dense(dims[-1], name=f'encoder_{n_stacks - 1}')(h)

    # internal layers in decoder
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name=f'decoder_{i}')(h)

    # output
    h = Dense(dims[0], name=f'decoder0')(h)

    return Model(inputs=x, outputs=h)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution

    # Example
    ```
        model.add(ClusteringLayer(n_clusters = 10))
    ```
    # Arguments
        n_clusters: number of clusters
        weights: list of Numpy array with shape `(n_clusters, n_features)` which represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0

    # Input Shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output Shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            (self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Student t-distribution, as same as used in t-sne algorithm.
            q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        :param inputs: variable containing data, shape=(n_samples, n_features)
        :return q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs,
                                                       axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256):
        super(DEC, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size
        self.autoencoder = autoencoder(self.dims)
        hidden = self.autoencoder.get_layer(
            name=f'encoder_{self.n_stacks - 1}').output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)

        # Prepare DEC Model
        clustering_layer = ClusteringLayer(
            self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=clustering_layer)
        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam'):
        print("...Pretraining...")
        # SGD(lr=0.01, momentum=0.9)
        self.autoencoder.compile(loss='mse', optimizer=optimizer)
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        self.autoencoder.save_weights('./SavedModels/ae_weights.h5')
        print("Pretrained Weights are saved to ./SavedModels/ae_weights.h5")
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of dec model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # Extract features from before clustering layer
        return self.encoder.predict(x)

    # predict cluster labels using the output of clustering layer
    def predict_clusters(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    # target dist P which enhances the discrimination of soft label q
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss='kld', optimizer='adam'):
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x, y=None, method='kmeans', batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None,
            save_dir='./results/dec'):
        print('Update interval:', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save Interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is not None:
            print("... Pretraining auteoncoders using default hyper-parameters:")
            print("    optimizer='adam'; epochs=200")
            self.pretrain(x, batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully')

        # Step 2: initialize cluster centers with k-means
        if method == 'kmeans':
            print('Initializing cluster centers with k-means.')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
            y_pred_last = np.copy(self.y_pred)
            self.model.get_layer(name='clustering').set_weights(
                [kmeans.cluster_centers_])

        elif method == 'hac':
            print('Initializing cluster centers with Agglomerative Clustering.')
            hac = AgglomerativeClustering(
                n_clusters=self.n_clusters, affinity='cosine', linkage='complete')
            self.y_pred = hac.fit_predict(X=x)
            centers = np.zeros((self.n_clusters, x.shape[-1]))
            for i in range(0, self.n_clusters):
                cluster_points = x[self.y_pred == i]
                cluster_mean = np.mean(cluster_points, axis=0)
                centers[i, :] = cluster_mean
            y_pred_last = np.copy(self.y_pred)
            self.model.get_layer(name='clustering').set_weights([centers])

        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(
            logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L'])
        logwriter.writeheader()

        loss = 0
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(
                    np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5)
                    nmi = np.round(
                        metrics.normalized_mutual_info_score(y, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss)
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi',
                          nmi, ', ari', ari, '; loss=', loss)

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                 y=p[index * self.batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                 y=p[index * self.batch_size:(index + 1) * self.batch_size])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to:', save_dir +
                      '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(
                    save_dir + '/DEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred
