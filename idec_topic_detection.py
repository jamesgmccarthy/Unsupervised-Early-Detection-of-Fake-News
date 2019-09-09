import time
import joblib
import os
import csv
import pandas as pd
import pandas as np
import sklearn.metrics as metrics
from gensim.models import Doc2Vec
from IDEC import IDEC
from TopicClustering import get_embeddings, create_tf_idf, get_median_pmi, get_most_common_words, word_count_of_corpus, \
    word_count_dict, bigram_count_dict


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


def load_data(dataset):
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


def main(dataset):
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
    data = load_data(dataset)
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


if __name__ == '__main__':
    main()
