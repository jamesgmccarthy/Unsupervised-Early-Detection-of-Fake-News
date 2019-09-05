"""""
Code is adapted from https://github.com/XifengGuo/IDEC/blob/master/IDEC.py
Implementation of Improved Deep Embedded Clustering as described in paper:
        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.
"""
from time import time
import numpy as np
import pandas as pd
import joblib
from keras.models import Model,  Sequential
import keras.layers as layers
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import os
import csv
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import keras
import keras.backend as K
from sklearn.preprocessing import normalize
from DEC_IDEC import cluster_acc, ClusteringLayer, dec_autoencoder
from gensim.models import Doc2Vec
from CreateEmbeddings import create_tagged_documents
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from TopicClustering import get_embeddings, create_tf_idf, get_median_pmi, get_most_common_words, word_count_of_corpus, \
    word_count_dict, bigram_count_dict
from keras.preprocessing import text


def reconstruction_loss(y_true, y_pred):
    """
    Define the reconstruction lose
    """

    y_true = K.l2_normalize(y_true, axis=-1)

    y_pred = K.l2_normalize(y_pred, axis=-1)

    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


class IDEC(object):
    def __init__(self,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 dataset='politifact'):

        super(IDEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.centers = []
        self.y_pred = []
        self.dataset = dataset

    def pretrain(self, x, save_autoencoder=True, batch_size=256, layerwise_pretrain_iters=50000, finetune_iters=100000, optimizer='adam', exp='fnd'):
        """
        adapted from https://github.com/nadavbar/DEC-Keras/blob/master/keras_dec.py
        """
        print('Greedy layer-wise pretraining...')
        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i in range(1, len(self.dims)):
            encoder_activation = 'linear' if i == (
                len(self.dims) - 1) else 'relu'  # linear if hidden layer
            # Initialise encoder layer, input is output of previous layer
            encoder = layers.Dense(self.dims[i], activation=encoder_activation,
                                   input_shape=(self.dims[i-1],),
                                   kernel_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros', name=f'encoder_dense_{i}')
            self.encoders.append(encoder)

            decoder_index = len(self.dims) - i
            decoder_activation = 'linear' if i == 1 else 'relu'  # linear if final layer
            # Initialise Decoder layer
            decoder = layers.Dense(self.dims[i-1], activation=decoder_activation,
                                   kernel_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                name=f'decoder_dense_{decoder_index}')
            self.decoders.append(decoder)

            autoencoder = Sequential([layers.Dropout(0.2, input_shape=(self.dims[i-1],), name=f'encoder_dropout_{i}'),
                                      encoder,
                                      layers.Dropout(
                                          0.2, name=f'decoder_dropout_{i}'),
                                      decoder])
            autoencoder.compile(loss='mse', optimizer=keras.optimizers.SGD(
                lr=0.1, decay=0, momentum=0.9))
            autoencoder.summary()
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder
        # Dropout is discarded
        self.encoder = Sequential(self.encoders)
        self.encoder.compile(loss='mse', optimizer=SGD(
            lr=0.1, decay=0, momentum=0.9))
        self.decoders.reverse()
        self.pretrain_autoencoder = Sequential(self.encoders+self.decoders)
        self.pretrain_autoencoder.compile(loss='mse', optimizer=SGD(
            lr=0.1, decay=0, momentum=0.9))
        iters_per_epoch = max(int(len(x)/batch_size), 1)
        layerwise_epochs = max(
            int(layerwise_pretrain_iters/iters_per_epoch), 1)
        finetune_epochs = max(int(finetune_iters / iters_per_epoch), 1)

        current_input = x
        lr_epoch_update = max(1, 2000/float(iters_per_epoch))

        def step_decay(epoch):
            initial_rate = 0.1
            factor = int(epoch/lr_epoch_update)
            lr = initial_rate/(10**factor)
            return lr

        lr_schedule = keras.callbacks.LearningRateScheduler(step_decay)
        # Train autoencoders in greedy layerwise fashion
        for i, autoencoder in enumerate(self.layer_wise_autoencoders):
            if i > 0:
                weights = self.encoders[i-1].get_weights()
                dense_layer = layers.Dense(self.dims[i], input_shape=(current_input.shape[1],),
                                           activation='relu', weights=weights, name=f'encoder_dense_copy{i}')
                encoder_model = Sequential([dense_layer])
                encoder_model.compile(loss='mse', optimizer=SGD(
                    lr=0.1, decay=0, momentum=0.9))
                current_input = encoder_model.predict(current_input)
            autoencoder.summary()
            autoencoder.fit(current_input, current_input,
                            batch_size=batch_size, epochs=layerwise_epochs, callbacks=[lr_schedule])

            # Set weights of end-to-end autoencoder
            self.pretrain_autoencoder.layers[i].set_weights(
                autoencoder.layers[1].get_weights())
            self.pretrain_autoencoder.layers[len(
                self.pretrain_autoencoder.layers)-i-1].set_weights(autoencoder.layers[-1].get_weights())

        if not os.path.exists(f'./SavedModels/idec/{self.dataset}/{exp}/'):
            os.makedirs(f'./SavedModels/idec/{self.dataset}/{exp}/')

        # Fine tune full autoencoder in reconstruction task
        print('Fine-tuning Autoencoder')
        self.pretrain_autoencoder.fit(
            x, x, batch_size=batch_size, epochs=finetune_epochs, callbacks=[lr_schedule])
        if save_autoencoder:
            self.pretrain_autoencoder.save_weights(
                f'./SavedModels/idec/{self.dataset}/{exp}/ae_weights.h5')
        self.autoencoder = dec_autoencoder(self.dims)

        # Inintialise IDEC Autoencoder
        self.autoencoder.load_weights(
            f'./SavedModels/idec/{self.dataset}/{exp}/ae_weights.h5')
        hidden = self.autoencoder.get_layer(
            name='encoder_%d' % (self.n_stacks-1)).output
        self.encoder = Model(inputs=self.autoencoder.input, outputs=hidden)
        clustering_layer = ClusteringLayer(
            self.n_clusters, alpha=self.alpha, name='clustering')(hidden)
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[clustering_layer, self.autoencoder.output])

        print(
            f'Pretrained weights are saved to ./SavedModels/idec/{self.dataset}/ae_weights.h5')
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    # predict cluster labels using the output of clustering layer
    def predict_clusters(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    # target distribution P which enhances the discrimination of soft label Q
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam'):
        self.model.compile(
            loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, method='kmeans', batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/idec', cluster=None, under_sample=False):

        print('Update inte  rval', update_interval)
        save_interval = update_interval + 1
        print('Save interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size=batch_size, method=method)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
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
                n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')
            x_pred = self.encoder.predict(x)
            self.y_pred = hac.fit_predict(x_pred)
            centers = np.zeros((self.n_clusters, x_pred.shape[-1]))
            for i in range(0, self.n_clusters):
                cluster_points = x_pred[self.y_pred == i]
                cluster_mean = np.mean(cluster_points, axis=0)
                centers[i, :] = cluster_mean
            y_pred_last = np.copy(self.y_pred)
            self.model.get_layer(name='clustering').set_weights([centers])

        # Step 3: deep clustering
        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'a')
        logwriter = csv.DictWriter(
            logfile, fieldnames=['dataset', 'iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(
                        metrics.normalized_mutual_info_score(y, self.y_pred, average_method='arithmetic'), 5)
                    ari = np.round(
                        metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(
                        dict(dataset=self.dataset, iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2]))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f, Lc= %.5f,  Lr= %.5f'
                          % (ite, acc, nmi, ari, loss[0], loss[1], loss[2]))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(
                    np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                print('saving model to: ' + save_dir +
                      'IDEC_model_' + str(ite) + '.h5')
                self.model.save_weights(
                    save_dir + 'IDEC_model_' + str(ite) + '.h5')

            ite += 1

        # save the trained model
        logfile.close()
        if under_sample is True:
            print('saving model to: ' + save_dir +
                  '/under_sampled_IDEC_model_final'+str(cluster)+'.h5')
            self.model.save_weights(
                save_dir + '/under_sampled_IDEC_model_final'+str(cluster)+'.h5')
        else:
            print('saving model to: ' + save_dir +
                  '/IDEC_model_final'+str(cluster)+'.h5')
            self.model.save_weights(
                save_dir + '/IDEC_model_final'+str(cluster)+'.h5')

        return self.y_pred


def classifier(X):
    """Basic Classifier set up in the same structure as the
    encoder part of the autoencoder

    Arguments:
        X {Numpy Array} -- Array of embeddings

    Returns:
        model {Keras Model} -- Classifier Model
    """
    input = layers.Input(shape=(X.shape[-1],))
    dense = layers.Dense(500, activation='relu')(input)
    dense_2 = layers.Dense(500, activation='relu')(dense)
    dense_3 = layers.Dense(2000, activation='relu')(dense_2)
    hidden = layers.Dense(10, activation='relu')(dense_3)
    dropout = layers.Dropout(rate=0.5)(hidden)
    output = layers.Dense(2, activation='softmax')(dropout)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def get_pmi_score(data, labels, num_words, word_type='stemmed_text'):
    """Calculate aggregate PMI score

    Arguments:
        data {Pandas DataFrame} -- DataFrame containing articles
        labels {NumPy Array} -- Cluster Assignments
        num_words {int} -- Number of words to calculate PMI score of
        word_type {string} -- Which word type to choose from dataframe i.e. stemmed_text/lemmatized_text

    Returns:
        agg_pmi - aggregate pmi score is the weighted mean of the median pmi score for each cluster in labels
    """
    data, wcd = word_count_dict(data, word_type)
    data, bigram_cd = bigram_count_dict(data, word_type)
    total_word_count = word_count_of_corpus(data, word_type)
    median_pmi_scores = {}
    agg_pmi = 0
    for label in np.unique(labels):
        common_words = get_most_common_words(
            data, label, num_words=num_words, text_type=word_type)
        if common_words is None:
            pass
        median_pmi_scores[label] = (get_median_pmi(list(common_words), wcd, bigram_cd, total_word_count),
                                    len(data[data['cluster'] == label]))
    for cluster, values in median_pmi_scores.items():
        if values[0] == float('-inf'):
            agg_pmi += 0
        else:
            agg_pmi += (values[0] * (values[1] / len(data)))
    return agg_pmi


def calculate_topic_metrics(data, df, y, num_words, num_topics, word_type, dataset):
    """Run topic clustering experiment and return metrics, cluster assignments and dataframe

    Arguments:
        data {Numpy Array} -- Array of embeddings
        df {Pandas DataFrame} -- DataFrame of data which experiment is being run on
        y {Numpy Array} -- Cluster Assignments
        num_words {int} -- Number of words to calculate PMI scores for
        num_topics {int} -- Numbers of topics to search for, i.e. number of clusters to assign data to
        word_type {str} -- Which word type to choose from dataframe i.e. stemmed_words/lemmatized_words
        dataset {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    batch_size = data.shape[0] // 10
    pretrain_epochs = 100
    optimizer = 'adam'
    update_interval = 100
    save_dir = f'./results/idec/TopicClustering/{dataset}/'
    idec = IDEC(dims=[data.shape[-1], 500, 500,
                      2000, 10], n_clusters=num_topics)
    t0 = time()

    # Greedy layerwise pretraining of autoencoder - Only Reconstruction
    idec.pretrain(data,
                  batch_size=batch_size, layerwise_pretrain_iters=5000, finetune_iters=10000, exp='topic')

    # Initialise IDEC Model
    idec.compile(loss=['kld', 'mse'], loss_weights=[
                 0.1, 1], optimizer=optimizer)
    idec.fit(data, y=y, method='kmeans', batch_size=batch_size, tol=0.0001, maxiter=1000,
             update_interval=update_interval, ae_weights=None, save_dir=save_dir)

    # Predict Input, extract embedded feature representation
    features = idec.extract_feature(data)

    # Calculate internal metrics
    dbs = metrics.cluster.davies_bouldin_score(
        features, idec.y_pred)
    sil = metrics.cluster.silhouette_score(
        features, idec.y_pred)

    labels = idec.y_pred
    df['cluster'] = labels

    # Calculate PMI Score
    pmi = get_pmi_score(df, labels, num_words=num_words, word_type=word_type)
    return dbs, sil, pmi, labels, df


def create_word_embeddings(data, max_len, max_num_words, embedding_size):
    """Return array of word embeddings of input data, data, to a given maximum length, max_len,
    pad with zeros if too small


    Arguments:
        data {Pandas DataFrame} -- Input dataframe
        max_len {int} -- Maximum number of words in document to be embedded, documents are truncated if longer than this, padded with zeros if shorter
        max_num_words {int} -- Maximum vocabulary size of tokenizer, will take most frequently occuring words
        embedding_size {int} -- Glove word embedding size (50/100/200/300)

    Returns:
        [array] -- array of word embeddings
    """
    words = data['text']
    tokenizer = text.Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(words)
    sequences = tokenizer.texts_to_sequences(words)
    word_index = tokenizer.word_index
    # pad all sequences to same length as max length
    word_data = keras.preprocessing.sequence.pad_sequences(sequences,
                                                           maxlen=max_len, padding='post', truncating='post')

    # Create Keras Embedding layer with pretrained Glove weights
    glove_layer = create_glove_embedding(
        embedding_size, max_num_words, tokenizer, max_len)

    # Create model to embed the data
    embedding_model = EMBED_MODEL(max_len, embed_size=embedding_size, max_words=max_num_words,
                                  embedding_layer=glove_layer)

    # Use embedding model to embed data, each embedding vector is averaged so
    word_embeddings = get_word_embedding(embedding_model, word_data)
    return word_embeddings


def EMBED_MODEL(max_len, embed_size, max_words, embedding_layer):
    """
    Create model with only embedding layer and output layer. Output layer is redundent, no input will be passed through model

    Arguments:
        max_len {int} -- Max length of input into embedding layer
        embed_size {int} -- Dimension size of embedding (i.e. 50/100/200/300 are in line with Glove embeddings)
        max_words {max_words} -- Maximum size of vocabulary
        embedding_layer {Keras Embedding Layer or None} -- Pre-made keras embeddin layer with Glove weights or None

    Returns:
        model {Keras Model} -- return model of embedding layer
        """
    encoded_input = layers.Input(
        shape=(max_len,), dtype='float32', name='encoded_input')
    if embedding_layer is None:
        embedding = layers.Embedding(output_dim=embed_size, input_dim=max_words,
                                     input_length=max_len, name='embedding_layer')(encoded_input)
    else:
        embedding = embedding_layer(encoded_input)
    output = layers.Dense(embed_size)(embedding)
    model = Model(inputs=encoded_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def get_word_embedding(model, encoded_text):
    """Returns mean of each embedded word within input text, encoded text

    Arguments:
        model {Keras Model} -- Embedding Model
        encoded_text {Keras Sequences} -- A keras tokenizer's text to sequence output

    Returns:
        [Array] -- Mean of word embeddings
    """
    embedding_layer_model = Model(
        inputs=model.input, outputs=model.get_layer('embedding_layer').output)
    return np.mean(embedding_layer_model.predict(encoded_text), axis=-1)


def create_glove_embedding(embedding_dim, max_num_words, tokenizer, max_seq_length):
    """Retrieve saved Glove embedding, create embedding matrix then return keras embedding
    layer with glove embedding maxtix saved as weights

    Arguments:
        embedding_dim {int} -- Dimension of glove embeddings
        max_num_words {int} -- Maximum size of vocab
        tokenizer {Keras Tokenizer} -- Tokenizer fitted to data
        max_seq_length {int} -- Maximum length of any input sequence

    Returns:
        [type] -- [description]
    """
    print('Pretrained embeddings GloVe is loading...')
    embedding_index = {}
    google_Drive = './drive/My Drive/Thesis/Embeddings/Glove/'
    f = open('./Glove/glove.6B/glove.6B.%id.txt' % embedding_dim)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embedding_index))
    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return layers.Embedding(input_dim=max_num_words, output_dim=embedding_dim,
                            input_length=max_seq_length, weights=[
                                embedding_matrix],
                            trainable=False,
                            name='embedding_layer')


def load_topic_data(dataset):
    if dataset == 'all':
        politifact = joblib.load(
            './Data/Preprocessed/politifact_clustering_large.h5')
        gossipcop = joblib.load(
            './Data/Preprocessed/gossipcop_clustering_large.h5')
        data = pd.DataFrame()
        for df in [politifact, gossipcop]:
            data = data.append(df)
    elif dataset == 'gossipcop':
        gossipcop = joblib.load(
            './Data/Preprocessed/gossipcop_clustering_large.h5')
        data = gossipcop
    else:
        politifact = joblib.load(
            './Data/Preprocessed/politifact_clustering_large.h5')
        data = pd.DataFrame()
        for df in [politifact, gossipcop]:
            data = data.append(df)

    return data


def run_topic_exp(dataset):
    """Run full topic experiment, following steps:

    - Load data
    - Create storage file for metrics
    - Loop through each partition
        - Load doc2vec embeddings
        - Create tf-idf vectors
        - Loop through each representation (embeddings/tf-idf)
            - Train idec model on representation
            - Calculate metrics
            - Save cluster assignments
            - Write results to metric file

    Arguments:
        dataset {str} -- The dataset to run experiment on (politifact or gossipcop)
    """
    print("loading Data")
    data = load_topic_data(dataset)
    metric_dict = {}  # Store all metrics to print out in terminal

    # Create file to record all the metrics
    if not os.path.isfile('./results/idec/TopicClustering/CSV/'):
        os.makedirs('./results/idec/TopicClustering/CSV/')
    metric_results_file = open(
        f'./results/idec/TopicClustering/CSV/{dataset}_clustering_scores.csv', 'a')
    logwriter = csv.DictWriter(metric_results_file,
                               fieldnames=['representation', 'number_of_topics', 'Davies_Bouldin_Score',
                                           'Silhouette_Score', 'PMI'])
    logwriter.writeheader()

    # Loop through each topic partition
    for num in [5, 10, 15]:
        num_topics = num
        num_words = 10
        word_type = 'stemmed_text'

        # Create doc2vec embeddings
        print('Loading Doc2vec model now')
        doc2vec = Doc2Vec.load(
            './SavedModels/saved_doc2vec_eval_model_clustering')
        embeddings = get_embeddings(data, doc2vec)

        # Create tf-idf embeddings
        print("Creating tf-idf Representations")
        one_gram, bi_gram, tri_gram = create_tf_idf(
            data, num_words=2000, word_type=word_type, force=True)
        one_gram = one_gram.todense()
        del bi_gram, tri_gram
        representations_dict = {
            'embeddings': embeddings, 'one_gram': one_gram}

        # Loop through each representations, train idec on data and calculate metrics
        for name, representations in representations_dict.items():
            dbs, sil, pmi, labels, df = calculate_topic_metrics(data=representations, df=data, y=None,
                                                                num_words=num_words, num_topics=num,
                                                                word_type=word_type, dataset=dataset)
            score_dict[f'Davies Bouldin Score {name} {num} '] = dbs
            score_dict[f'Silhouette Score {name} {num}:'] = sil
            score_dict[f'PMI Score {name} {num}:'] = pmi

            # save cluster assignments of idec
            joblib.dump(
                df, f'./results/idec/TopicClustering/{dataset}_idec_labels_{name}_{num}.h5')

            # Write results to metric file
            logwriter.writerow(dict(representation=name, number_of_topics=num_topics, Davies_Bouldin_Score=dbs,
                                    Silhouette_Score=sil, PMI=pmi))

    print(metric_dict)


def run_fnd_experiment(dataset='politifact', topics=False, under_sample=False):
    print("loading Data")
    if dataset == 'gossipcop' and topics is not True:
        gossipcop = joblib.load(
            './Data/Preprocessed/gossipcop_fnd_large.h5')
        data = gossipcop
        hcfs = joblib.load('./Data/HandCraftedFeatures/gossipcop.h5')

    elif dataset == 'politifact' and topics is True:
        df = joblib.load(
            './results/politifact/TopicClustering/lda_topic_data_5.h5')
        hcf = joblib.load('./Data/HandCraftedFeatures/politifact_large.h5')
        dataset = 'topicsPolitifact'

    elif dataset == 'gossipcop' and topics is True:
        df = joblib.load(
            './results/gossipcop/TopicClustering/lda_topic_data_5.h5')
        hcf = joblib.load('./Data/HandCraftedFeatures/gossipcop.h5')
        dataset = 'topicsGossipcop'

    else:
        politifact = joblib.load(
            './Data/Preprocessed/politifact_fnd_large.h5')
        gossipcop = joblib.load(
            './Data/preprocessed/gossipcop_clustering_large.h5')
        data = pd.DataFrame()
        for df in [politifact, gossipcop]:
            data = data.append(df)
        hcfs = joblib.load('./Data/HandCraftedFeatures/politifact_large.h5')

    # To control number of clusters idec model will be trained on, when 1 idec trained on full dataset
    if dataset == 'topicsPolitifact' or dataset == 'topicsGossipcop':
            topic_cluster_num = len(np.unique(df['cluster']))
    else:
        topic_cluster_num = 1

    for i in range(0, topic_cluster_num):
            if topic_cluster_num > 1:
                data = df[df['cluster'] == i]
                print("Running exp on cluster:", i)
                hcfs = hcf[hcf['cluster'] == i]

                
def main(exp, dataset='politifact', topics=False, under_sample=False):
    if not os.path.exists('results'):
        os.makedirs('results')

    if exp == 'fnd':

        """
        Clustering Experiment
        """

        if dataset == 'topicsPolitifact' or dataset == 'topicsGossipcop':
            topic_cluster_num = len(np.unique(df['cluster']))
        else:
            topic_cluster_num = 1
        for i in range(0, topic_cluster_num):
            if topic_cluster_num > 1:
                data = df[df['cluster'] == i]
                print("Running exp on cluster:", i)
                hcfs = hcf[hcf['cluster'] == i]
            # Doc2vec data
            doc2vec = Doc2Vec.load(
                './SavedModels/saved_doc2vec_eval_model_fnd')
            print("Creating Tagged Docs")
            if under_sample is True:
                # 1 indicates fake news
                fake_sample_size = len(data[data.label == 1])
                fake = data[data.label == 1]
                real_indices = data[data.label == 0].index
                random_real_indices = np.random.choice(
                    real_indices, fake_sample_size + 1, replace=False)
                real_undersample_set = data.loc[random_real_indices]
                data = fake.append(real_undersample_set)

            training_data = create_tagged_documents(data)
            x = np.array([doc2vec.infer_vector(doc.words, epochs=50, alpha=0.01, min_alpha=0.0001)
                          for doc in training_data])

            print(x.shape)
            y = data['label'].values
            # word_embeddings= create_word_embeddings(data,1000,150000,300)
            # print(word_embeddings.shape)
            # HandCrafted Features
            y_hcf = hcfs['label'].values
            hcfs.drop('label', axis=1, inplace=True)
            hcfs = hcfs.values
            hcfs = normalize(hcfs)
            kmeans = KMeans(n_clusters=2, n_init=20)
            hcf_y_pred = kmeans.fit_predict(hcfs)
            acc = cluster_acc(y_hcf, hcf_y_pred)
            nmi = metrics.normalized_mutual_info_score(
                y_hcf, hcf_y_pred, average_method='arithmetic')
            print('nmi:', nmi)
            adj = metrics.adjusted_rand_score(y_hcf, hcf_y_pred)
            print('ARI:', adj)
            if y is not None:
                if under_sample == True:
                    fake_news_results_file = open(
                        f'./results/FakeNews/CSV/{dataset}_Undersampled_FN_Detection_doc2vec.csv', 'a')
                elif under_sample == False:
                    fake_news_results_file = open(
                        f'./results/FakeNews/CSV/{dataset}FN_Detection_doc2vec.csv', 'a')
                logwriter = csv.DictWriter(fake_news_results_file,
                                           fieldnames=['Method', 'ClusteringACC', 'NMI', 'ADJ'])
                logwriter.writeheader()
            # Writing HCF results
            logwriter.writerow(
                dict(Method='KMeans HCF', ClusteringACC=acc, NMI=nmi, ADJ=adj))

            print('Fitting kmeans to doc2vec')
            kmeans = KMeans(n_clusters=2, n_init=20)
            # dist = 1-cosine_similarity(x)
            y_pred = kmeans.fit_predict(x)
            acc = cluster_acc(y, y_pred)
            print('Clustering Acc:', acc)
            nmi = metrics.normalized_mutual_info_score(
                y, y_pred, average_method='arithmetic')
            print('nmi:', nmi)
            adj = metrics.adjusted_rand_score(y, y_pred)
            print('ARI:', adj)
            logwriter.writerow(dict(Method='KMeans Doc2vec',
                                    ClusteringACC=acc, NMI=nmi, ADJ=adj))
            tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
            print('True Neg:', tn)
            print("False Positive:", fp)
            print('False Negative:', fn)
            print("True Positive:", tp)
            f1 = metrics.classification.f1_score(y, y_pred)
            acc = metrics.accuracy_score(y, y_pred)
            recall_s = metrics.recall_score(y, y_pred)
            precision_s = metrics.precision_score(y, y_pred)
            print("Original Score for doc2vec vectors")
            print('f1_score:', f1, 'accuracy:',
                  acc, 'recall:', recall_s, 'precision_s', precision_s)

            # Run IDEC Experiment
            batch_size = 256
            pretrain_epochs = 200
            optimizer = 'adam'
            update_interval = 140
            save_interval = 10
            save_dir = f'./results/idec/{dataset}_{exp}_dDoc2vec'
            n_clusters = 2

            idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10],
                        n_clusters=n_clusters, dataset=dataset)
            print("Running IDEC Experiment")
            idec.pretrain(x, layerwise_pretrain_iters=50000,
                          finetune_iters=100000, batch_size=batch_size, exp='fnd')
            plot_model(idec.model, to_file='idec_model.png', show_shapes=True)
            idec.autoencoder.summary()
            idec.model.summary()
            idec.compile(loss=['kld', 'mse'], loss_weights=[
                1, 0.1], optimizer=optimizer)
            idec.fit(x, y=y, method='kmeans', batch_size=batch_size, tol=0.0001, maxiter=100000,
                     update_interval=update_interval,
                     ae_weights=None, save_dir=save_dir, cluster=i, under_sample=under_sample)
            extracted_x = idec.extract_feature(x)
            kmeans = KMeans(n_clusters=2, n_init=20)
            y_pred_extracted = kmeans.fit_predict(extracted_x)
            acc = cluster_acc(y, y_pred_extracted)
            nmi = metrics.normalized_mutual_info_score(
                y, y_pred_extracted, average_method='arithmetic')
            adj = metrics.adjusted_rand_score(y, y_pred_extracted)
            print(f'Extracted acc: {acc}, nmi: {nmi}, adj={adj}')
            logwriter.writerow(
                dict(Method='KMeans Extracted Features', ClusteringACC=acc, NMI=nmi, ADJ=adj))
            """
            print("Running IDEC Experiment with tf-idf features")
            tf_vect = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.8, min_df=0.05, ngram_range=(1, 3),
                                  max_features=50000)
            tf_idf = tf_vect.fit_transform(data['text'])
            tf_idf = tf_idf.todense()
            idec_tfidf = IDEC(dims=[tf_idf.shape[-1], 500, 500, 2000, 10],n_clusters=n_clusters,dataset=dataset)
            idec_tfidf.pretrain(tf_idf,layerwise_pretrain_iters=5000,finetune_iters=10000, batch_size=batch_size,exp='fnd')
            idec_tfidf.compile(loss=['kld','mse'],loss_weights=[0.1,1],optimizer=optimizer)
            idec_tfidf.fit(tf_idf, y=y, method='kmeans', batch_size=batch_size, tol=0.0001, maxiter=100000,
                    update_interval=update_interval,
                    ae_weights=None, save_dir=save_dir,cluster=i,under_sample= under_sample)
            tf_idf_extracted_x = idec_tfidf.extract_feature(tf_idf)
            kmeans = KMeans(n_clusters = 2,n_init=20)
            y_pred_extracted_tfidf = kmeans.fit_predict(tf_idf_extracted_x)
            acc = cluster_acc(y,y_pred_extracted_tfidf)
            nmi =metrics.normalized_mutual_info_score(y,y_pred_extracted_tfidf,average_method='arithmetic')
            adj=metrics.adjusted_rand_score(y,y_pred_extracted_tfidf)
            print(f'Extracted acc: {acc}, nmi: {nmi}, adj={adj}')
            logwriter.writerow(dict(Method= 'KMeans Extracted Features TFIDF', ClusteringACC=acc,NMI=nmi,ADJ=adj))
        
            """

            if y is not None:
                # Run Classification Experiments
                fake_news_results_file_classification = open(
                    './results/FakeNews/CSV/FN_Detection_doc2vec_classification', 'a')

                logwriter_cf = csv.DictWriter(fake_news_results_file,
                                              fieldnames=['Method', 'ACC', 'F1', 'Recall', 'Precision'])
                logwriter_cf.writeheader()
                y_split = keras.utils.to_categorical(y)
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y_split, test_size=0.25, random_state=0)
                cf = classifier(X_train)
                early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0.01, patience=10,
                                                              restore_best_weights=True)
                cf.fit(X_train, y_train, batch_size=256, epochs=100,
                       validation_split=0.1, callbacks=[early_stopper])
                y_pred = cf.predict(X_test)
                y_pred = y_pred.argmax(1)
                y_test = y_test.argmax(1)
                acc_score = metrics.accuracy_score(y_test, y_pred)
                f1 = metrics.f1_score(y_test, y_pred)
                recall_s = metrics.recall_score(y_test, y_pred)
                precision_s = metrics.precision_score(y_test, y_pred)
                print("Acc of cf: ", acc_score)
                print('F1 of cf: ', f1)
                print("Recall of cf:", recall_s)
                print("Precision of cf:", precision_s)
                logwriter_cf.writerow(dict(
                    Method='Classification Full', ACC=acc_score, F1=f1, Recall=recall_s, Precision=precision_s))
                print("Testing idec.y_pred")
                """
                tn, fp, fn, tp = metrics.confusion_matrix(y, idec.y_pred).ravel()

                print("learned tn:", tn)
                print("learned fp:", fp)
                print("learned fn:", fn)
                print("learned tp:", tp)
                f1 = metrics.classification.f1_score(y, idec.y_pred)
                acc_score = metrics.accuracy_score(y, idec.y_pred)
                recall_s = metrics.recall_score(y, idec.y_pred)
                precision_s = metrics.precision_score(y, idec.y_pred)
                print('f1_score:', f1, 'accuracy:',
                    acc_score, 'recall:', recall_s, 'precision_s', precision_s)
                logwriter_cf.writerow(
                    dict(Method='Clustering_clss', ACC=acc_score,F1=f1,Recall=recall_s,Precision=precision_s))
                print("Testing encoder_predictions")
                X_features = idec.extract_feature(x)
                kmeans = KMeans(n_clusters=2, n_init=20)
                y_pred = kmeans.fit_predict(X_features)
                tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
                print("test tn:", tn)
                print('test fp:', fp)
                print('test fn:', fn)
                print('test tp:', tp)
                f1 = metrics.classification.f1_score(y, y_pred)
                acc = metrics.accuracy_score(y, y_pred)
                recall_s = metrics.recall_score(y, y_pred)
                precision_s = metrics.precision_score(y, y_pred)
                print('f1_score:', f1, 'accuracy:',
                    acc, 'recall:', recall_s, 'precision_s', precision_s)
                logwriter_cf.writerow(
                    dict(Method='KMeans of features', ACC=acc_score,F1=f1,Recall=recall_s,Precision=precision_s))
                print("now testing idec features on classifier")
                X_train_idec_features = idec.extract_feature(X_train)
                X_test_idec_features = idec.extract_feature(X_test)
                idec_classifier = classifier(X_train_idec_features)
                idec_classifier.fit(X_train_idec_features, y_train, batch_size=256, epochs=100, validation_split=0.1,
                                    callbacks=[early_stopper])

                y_idec_pred = idec_classifier.predict(X_test_idec_features)
                y_idec_pred = y_idec_pred.argmax(1)
                tn, fp, fn, tp = metrics.confusion_matrix(
                    y_test, y_idec_pred).ravel()
                print("test tn:", tn)
                print('test fp:', fp)
                print('test fn:', fn)
                print('test tp:', tp)
                f1 = metrics.classification.f1_score(y_test, y_idec_pred)
                acc = metrics.accuracy_score(y_test, y_idec_pred)
                recall_s = metrics.recall_score(y_test, y_idec_pred)
                precision_s = metrics.precision_score(y_test, y_idec_pred)
                print('f1_score:', f1, 'accuracy:',
                    acc, 'recall:', recall_s, 'precision_s', precision_s)
                logwriter_cf.writerow(
                    dict(Method='Classifier_idec_features', ACC=acc_score,F1=f1,Recall=recall_s,Precision=precision_s))

                """


if __name__ == '__main__':
    main(exp='fnd', dataset='gossipcop', under_sample=False, topics=False)
