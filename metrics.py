from gensim.models import Doc2Vec
from IDEC import IDEC
import joblib
import numpy as np
from TopicClustering import create_tagged_documents
from DEC_IDEC import cluster_acc, ClusteringLayer, dec_autoencoder
from keras.models import Model
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import csv
import os
import coclust.evaluation.external as external


def load_embeddings(data):
    doc2vec = Doc2Vec.load('./SavedModels/saved_doc2vec_eval_model_fnd')
    training_data = create_tagged_documents(data)
    x = np.array([doc2vec.infer_vector(doc.words, epochs=50,
                                       alpha=0.01, min_alpha=0.0001) for doc in training_data])
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


def predict_labels(x, idec):
    features = idec.extract_feature(x)
    kmeans = KMeans(n_clusters=2, n_init=20)
    y_pred = kmeans.fit_predict(features)
    return y_pred, features


def eval_labels(x, y, labels):
    # acc = cluster_acc(y, labels)
    acc = external.accuracy(y, labels)
    nmi = metrics.adjusted_mutual_info_score(
        y, labels, average_method='geometric')
    adj = metrics.adjusted_rand_score(y, labels)
    sil = metrics.silhouette_score(x, labels)
    db = metrics.davies_bouldin_score(x, labels)
    fm = metrics.fowlkes_mallows_score(y, labels)
    cont_matrix = metrics.cluster.contingency_matrix(labels, y)
    print(cont_matrix)
    print("acc", acc)
    print('nmi', nmi)
    print('adj', adj)
    print('sil', sil)
    print('db', db)
    print('fm:', fm)
    return acc, nmi, adj, sil, db, fm


def main(under_sample=False):
    politifact = joblib.load(
        './results/politifact/TopicClustering/lda_topic_data_5.h5')
    gossipcop = joblib.load(
        './results/gossipcop/TopicClustering/lda_topic_data_5.h5')

    if not os.path.exists('./results/FakeNews/Metrics'):
        os.makedirs('./results/FakeNews/Metrics')
    datasets = {'gossipcop': gossipcop}
    for name, df in datasets.items():
        """
        if name == 'gossipcop':
            hcf = joblib.load('./Data/HandCraftedFeatures/gossipcop.h5')
        elif name == 'politifact':
            hcf = joblib.load('./Data/HandCraftedFeatures/politifact_large.h5')
        """
        # undersample data if true
        if under_sample is True:
            # 1 indicates fake news
            fake_sample_size = len(df[df.label == 1])
            fake = df[df.label == 1]
            real_indices = df[df.label == 0].index
            random_real_indices = np.random.choice(
                real_indices, fake_sample_size + 1, replace=False)
            real_undersample_set = df.loc[random_real_indices]
            df_undersampled = fake.append(real_undersample_set)
            # Extracted feature test
            x, y = load_embeddings(df_undersampled)

        elif under_sample is False:
            x, y = load_embeddings(df)

        idec = create_model(x, dataset=name, topics=False,
                            under_sample=under_sample)
        y_pred, features = predict_labels(x, idec)
        acc, nmi, adj, sil, db, fm = eval_labels(features, y, y_pred)
        # Create full dataset csv
        if under_sample == True:
            full_file = open(
                f'./results/FakeNews/Metrics/under_sampled_{name}_full.csv', 'a')
            full_logwrite = csv.DictWriter(full_file, fieldnames=[
                'Dataset', 'ClusteringAcc', 'NMI', 'ARI', 'FM', 'Silhouette', 'Davies_Bouldin'])
        elif under_sample == False:
            full_file = open(
                f'./results/FakeNews/Metrics/{name}_full.csv', 'a')
            full_logwrite = csv.DictWriter(full_file, fieldnames=[
                'Dataset', 'ClusteringAcc', 'NMI', 'ARI', 'FM', 'Silhouette', 'Davies_Bouldin', ])
        full_logwrite.writeheader()
        full_logwrite.writerow(dict(Dataset='Extracted', ClusteringAcc=acc,
                                    NMI=nmi, ARI=adj, FM=fm, Silhouette=sil, Davies_Bouldin=db))
        # Doc2vec Test
        kmeans = KMeans(n_clusters=2, n_init=20)
        y_pred = kmeans.fit_predict(x)
        print(x.shape)
        acc, nmi, adj, sil, db, fm = eval_labels(x, y, y_pred)
        full_logwrite.writerow(dict(Dataset='Doc2vec', ClusteringAcc=acc,
                                    NMI=nmi, ARI=adj, FM=fm, Silhouette=sil, Davies_Bouldin=db))
        """
        # HCF Test
        if under_sample is True:
            # 1 indicates fake news
            fake_sample_size = len(hcf[hcf.label == 1])
            fake = hcf[hcf.label == 1]
            real_indices = hcf[hcf.label == 0].index
            random_real_indices = np.random.choice(
                real_indices, fake_sample_size + 1, replace=False)
            real_undersample_set = hcf.loc[random_real_indices]
            hcf = fake.append(real_undersample_set)
        y = hcf['label'].values
        hcf_topic = hcf.copy()
        hcf.drop(['label', 'cluster'], axis=1, inplace=True)
        kmeans = KMeans(n_clusters=2, n_init=20)
        y_pred = kmeans.fit_predict(hcf)
        acc, nmi, adj, sil, db, fm = eval_labels(hcf, y, y_pred)
        full_logwrite.writerow(dict(Dataset='HCF', ClusteringAcc=acc,
                                    NMI=nmi, ARI=adj, FM=fm, Silhouette=sil, Davies_Bouldin=db))
        """
        # Create split csv
        if under_sample == True:
            split_file = open(
                f'./results/FakeNews/Metrics/under_sample_{name}_split.csv', 'a')
            split_logwrite = csv.DictWriter(split_file, fieldnames=[
                'Dataset', 'Cluster', 'Cluster_len', 'Fake_Articles', 'ClusteringAcc', 'NMI', 'ARI', 'FM', 'Silhouette',
                'Davies_Bouldin'])
        elif under_sample == False:
            split_file = open(
                f'./results/FakeNews/Metrics/{name}_split.csv', 'a')
            split_logwriter = csv.DictWriter(split_file, fieldnames=[
                'Dataset', 'Cluster', 'Cluster_len', 'Fake_Articles', 'ClusteringAcc', 'NMI', 'ARI', 'FM', 'Silhouette',
                'Davies_Bouldin'])
        split_logwriter.writeheader()
        for i in range(0, len(np.unique(df['cluster']))):
            if under_sample is True:
                # 1 indicates fake news
                fake_sample_size = len(df[(df.label == 1) & (df.cluster == i)])
                fake = df[(df.label == 1) & (df.cluster == i)]
                real_indices = df[(df.label == 0) & (df.cluster == i)].index
                random_real_indices = np.random.choice(
                    real_indices, fake_sample_size + 1, replace=False)
                real_undersample_set = df.loc[random_real_indices]
                df_undersampled = fake.append(real_undersample_set)
                # Extracted feature test
                x, y = load_embeddings(df_undersampled)

            elif under_sample is False:
                # Create doc2vec embeddings
                x, y = load_embeddings(df[df['cluster'] == i])

            # IDEC feature test
            idec = create_model(x, dataset=name, topics=True,
                                cluster=i, under_sample=under_sample)
            y_pred, features = predict_labels(x, idec)
            acc, nmi, adj, sil, db, fm = eval_labels(features, y, y_pred)
            fk = len(df[(df['label'] == 1) & (df['cluster'] == i)])
            print(fk)
            split_logwriter.writerow(
                dict(Dataset='Extracted', Cluster=i, Cluster_len=len(x), Fake_Articles=fk, ClusteringAcc=acc,
                     NMI=nmi, ARI=adj, FM=fm, Silhouette=sil, Davies_Bouldin=db))
            # Doc2vec test
            kmeans = KMeans(n_clusters=2, n_init=20, init='random')
            y_pred = kmeans.fit_predict(x)
            acc, nmi, adj, sil, db, fm = eval_labels(x, y, y_pred)
            split_logwriter.writerow(
                dict(Dataset='Doc2vec', Cluster=i, Cluster_len=len(x), Fake_Articles=fk, ClusteringAcc=acc,
                     NMI=nmi, ARI=adj, FM=fm, Silhouette=sil, Davies_Bouldin=db))
            """
            # HCF test
            hcf = hcf_topic[hcf_topic['cluster'] == i]
            y_hcf = hcf['label'].values
            hcf.drop(['label', 'cluster'], axis=1, inplace=True)
            kmeans = KMeans(n_clusters=2, n_init=20, init='random')
            y_pred = kmeans.fit_predict(hcf)
            acc, nmi, adj, sil, db, fm = eval_labels(hcf, y_hcf, y_pred)
            split_logwriter.writerow(
                dict(Dataset='HCF', Cluster=i, Cluster_len=len(x), Fake_Articles=fk, ClusteringAcc=acc,
                     NMI=nmi, ARI=adj, Silhouette=sil, Davies_Bouldin=db))
            """


if __name__ == '__main__':
    main(under_sample=True)
