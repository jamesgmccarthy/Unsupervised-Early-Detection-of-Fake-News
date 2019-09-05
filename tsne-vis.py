'''Part of code adapted from https://github.com/XifengGuo/IDEC/issues/1
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from IDEC import IDEC
import joblib
from gensim.models import Doc2Vec
import numpy as np
from TopicClustering import create_tagged_documents
from sklearn.preprocessing import normalize
from DEC_IDEC import cluster_acc, ClusteringLayer, dec_autoencoder
from keras.models import Model,  Sequential
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.metrics as metrics


def load_embeddings(data):
    doc2vec = Doc2Vec.load('./SavedModels/saved_doc2vec_eval_model_fnd')
    training_data = create_tagged_documents(data)
    x = np.array([doc2vec.infer_vector(doc.words, epochs=50, alpha=0.01, min_alpha=0.0001)
                  for doc in training_data])
    y = data['label'].values
    return x, y


def create_model(x, dataset, topics=False, cluster=None, under_sample=False):
    # Create Model
    idec = IDEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=2)
    idec.autoencoder = dec_autoencoder(idec.dims)
    hidden = idec.autoencoder.get_layer(
        name='encoder_%d' % (idec.n_stacks - 1)).output
    idec.encoder = Model(inputs=idec.autoencoder.input, outputs=hidden)

    # Prepare clustering layer and model
    clustering_layer = ClusteringLayer(
        idec.n_clusters, alpha=idec.alpha, name='clustering')(hidden)
    idec.model = Model(inputs=idec.autoencoder.input,
                       outputs=[clustering_layer, idec.autoencoder.output])
    idec.model.summary()

    # Load pretrained weights
    if topics == False and under_sample == False:
        print(
            f"Loading weights from './results/idec/{dataset}_fnd_dDoc2vec/IDEC_model_final0.h5'")
        idec.load_weights(
            f'./results/idec/{dataset}_fnd_dDoc2vec/IDEC_model_final0.h5')
    if topics == True and under_sample == False:
        print(
            f"Loading weights from './results/idec/topics{dataset}_fnd_dDoc2vec/IDEC_model_final{cluster}.h5'")
        idec.load_weights(
            f'./results/idec/topics{dataset}_fnd_dDoc2vec/IDEC_model_final{cluster}.h5')
    if topics == False and under_sample == True:
        print(
            f"Loading weights from './results/idec/{dataset}_fnd_dDoc2vec/under_sampled_IDEC_model_final0.h5'")
        idec.load_weights(
            f'./results/idec/{dataset}_fnd_dDoc2vec/under_sampled_IDEC_model_final0.h5')
    if topics == True and under_sample == True:
        print(
            f"Loading weights from './results/idec/topics{dataset}_fnd_dDoc2vec/IDEC_model_final{cluster}.h5'")
        idec.load_weights(
            f'./results/idec/topics{dataset}_fnd_dDoc2vec/under_sampled_IDEC_model_final{cluster}.h5')
    return idec


def create_pca_points(features, tsne):
    pca = PCA(n_components=5)
    pca_results = pca.fit_transform(features)

    pca_embed_points = tsne.fit_transform(pca_results)
    return pca_embed_points


def plot_tsne(features, path, y):
    fig = plt.figure()
    colors = ['black', 'red']
    marker = ['x', '+']
    for i in range(2):
        plt.scatter(features[y == i, 0], features[y ==
                                                  i, 1], c=colors[i], marker=marker[i], label=str(i))
        plt.xticks(())
        plt.yticks(())
    fig.savefig(path+'fig-tsne.pdf', dpi=600)
    fig.clf()
    plt.clf()
    plt.close(fig)


def main(under_sample=False):
    gossipcop = (joblib.load(
        './results/gossipcop/TopicClustering/lda_topic_data_5.h5'), 'gossipcop')
    politifact = (joblib.load(
        './results/politifact/TopicClustering/lda_topic_data_5.h5'), 'politifact')
    for df in [gossipcop, politifact]:
        if under_sample is True:
            # 1 indicates fake news
            fake_sample_size = len(df[0][df[0].label == 1])
            fake = df[0][df[0].label == 1]
            real_indices = df[0][df[0].label == 0].index
            random_real_indices = np.random.choice(
                real_indices, fake_sample_size + 1, replace=False)
            real_undersample_set = df[0].loc[random_real_indices]
            df_temp = (fake.append(real_undersample_set), df[1])
        elif under_sample is False:
            df_temp = df
        x, y = load_embeddings(df_temp[0])
        idec = create_model(x, dataset=df_temp[1], under_sample=under_sample)
        features = idec.extract_feature(x)
        tsne = TSNE(n_components=2, verbose=1, n_iter=5000,
                    learning_rate=10, perplexity=30)
        doc_points = tsne.fit_transform(x)
        embed_points = tsne.fit_transform(features)
        pca_embed_points = create_pca_points(features, tsne=tsne)
        path = './TSNE_vis/'+df[1]+'/fulldataset/doc2vec'
        plot_tsne(doc_points, path=path, y=y)
        path = './TSNE_vis/'+df[1]+'/fulldataset/pca'
        plot_tsne(pca_embed_points, path=path, y=y)
        path = './TSNE_vis/'+df[1]+'/fulldataset/full'
        plot_tsne(embed_points, path=path, y=y)

    # Topic Split data
    for df in [gossipcop, politifact]:
        for cluster in range(0, 5):
            data = df[0][df[0]['cluster'] == cluster]
            x, y = load_embeddings(data)
            idec = create_model(x, dataset=df[1], topics=True, cluster=cluster)
            tsne = TSNE(n_components=2, verbose=1, n_iter=5000)
            features = idec.extract_feature(x)
            doc_points = tsne.fit_transform(x)
            embed_points = tsne.fit_transform(features)
            pca_embed_points = create_pca_points(features, tsne=tsne)
            path = './TSNE_vis/' + df[1]+'/fulldataset/doc2vec'+str(cluster)
            plot_tsne(doc_points, path=path, y=y)
            path = './TSNE_vis/' + df[1]+'/split/pca'+str(cluster)
            plot_tsne(pca_embed_points, path=path, y=y)
            path = './TSNE_vis/' + df[1]+'/fulldataset/full'+str(cluster)
            plot_tsne(embed_points, path, y=y)


main(under_sample=True)
