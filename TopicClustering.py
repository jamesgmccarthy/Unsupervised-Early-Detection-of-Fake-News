"""
Topic Detection phase of project
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from gensim.models import Doc2Vec
from CreateEmbeddings import create_tagged_documents
import matplotlib.pyplot as plt
from nltk import ngrams
import os
import csv
import pyreadr
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.phrases import npmi_scorer


def get_embeddings(data, model):
    """Returns doc2vec embeddings of input data using given model

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        model {Doc2vec Model} -- Chosen Doc2vec model to infer embeddings

    Returns:
        [Numpy Array] -- Array of embeddings
    """
    print("Creating Tagged Docs")
    training_data = create_tagged_documents(data)
    print("Infering Vectors")
    embeddings = np.array([model.infer_vector(doc.words, epochs=50, alpha=0.01, min_alpha=0.0001)
                           for doc in training_data])
    return embeddings


def create_tf_idf(data, num_words, text_type='stemmed_text', force=False):
    """Returns tf-idf matrices of uni-grams, bi-grams, tri-grams of input data

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_words {int} -- Threshold controlling the number of highest frequency words to use

    Keyword Arguments:
        text_type {str} -- Chosen word type for input into tf-idf matrices (default: {'stemmed_text'})
        force {bool} --  Creates new matrices if force is True, loads pre-made matrices if force is False (default: {False})

    Returns:
        [Sparse Matrices] -- uni-gram, bi-gram and tri-gram sprase tf-idf matrices 
    """
    if force == False:
        if os.path.isfile("./Embeddings/one_gram_tfidf_alldata.h5"):
            uni_gram = joblib.load('./Embeddings/uni_gram_tfidf_alldata.h5')
        if os.path.isfile('./Embeddings/bi_gram_tfidf_alldata.h5'):
            bi_gram = joblib.load('./Embeddings/bi_gram_tfidf_alldata.h5')
        if os.path.isfile('./Embeddings/tri_gram_tfidf_alldata.h5'):
            tri_gram = joblib.load('./Embeddings/tri_gram_tfidf_alldata.h5')
    else:
        tf_vect = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.8, min_df=0.05, ngram_range=(1, 1),
                                  max_features=num_words)
        bigram_tf_vect = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.95, min_df=0.05,
                                         ngram_range=(2, 2),
                                         max_features=num_words)
        trigram_tf_vect = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.95, min_df=0.05,
                                          ngram_range=(3, 3),
                                          max_features=num_words)

        print("Creating one-gram matrix")
        uni_gram = tf_vect.fit_transform(data[text_type])
        print("Creating bi-gram matrix")
        bi_gram = bigram_tf_vect.fit_transform(data[text_type])
        print("Creating tri-gram matrix")
        tri_gram = trigram_tf_vect.fit_transform(data[text_type])

        joblib.dump(uni_gram, './Embeddings/uni_gram_tfidf_alldata.h5')
        joblib.dump(bi_gram, './Embeddings/bi_gram_tfidf_alldata.h5')
        joblib.dump(tri_gram, './Embeddings/tri_gram_tfidf_alldata.h5')

    return uni_gram, bi_gram, tri_gram


def create_bow_representations(data, num_words, text_type='stemmed_text', force=False):
    """Return uni-gram, bi-gram and tri-gram bag of word matrices for input data

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_words {int} -- Threshold controlling the number of highest frequency words to use

    Keyword Arguments:
        text_type {str} -- Chosen word type for input into bag of word matrices (default: {'stemmed_text'})
        force {bool} --  Creates new matrices if force is True, loads pre-made matrices if force is False (default: {False})

    Returns:
        [Sparse Matrices] -- uni-gram, bi-gram and tri-gram sprase bag of word matrices 
    """
    if force == False:
        if os.path.isfile("./Embeddings/one_gram_bow_alldata.h5"):
            one_gram = joblib.load('./Embeddings/one_gram_bow_alldata.h5')
        if os.path.isfile('./Embeddings/bi_gram_bow_alldata.h5'):
            bi_gram = joblib.load('./Embeddings/bi_gram_bow_alldata.h5')
        if os.path.isfile('./Embeddings/tri_gram_bow_alldata.h5'):
            tri_gram = joblib.load('./Embeddings/tri_gram_bow_alldata.h5')
    one_gram_bow = CountVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.8, min_df=0.05,
                                   max_features=num_words,
                                   ngram_range=(1, 1))
    bi_gram_bow = CountVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.95, min_df=0.05,
                                  max_features=num_words,
                                  ngram_range=(2, 2))
    tri_gram_bow = CountVectorizer(tokenizer=lambda x: word_tokenize(x), max_df=0.95, min_df=0.05,
                                   max_features=num_words,
                                   ngram_range=(3, 3))

    print("Creating one-gram BOW matrix")
    one_gram = one_gram_bow.fit_transform(data[text_type])
    print("Creating bi-gram BOW matrix")
    bi_gram = bi_gram_bow.fit_transform(data[text_type])
    print('Creating tri-gram BOW matrix')
    tri_gram = tri_gram_bow.fit_transform(data[text_type])

    joblib.dump(one_gram, './Embeddings/one_gram_bow_alldata.h5')
    joblib.dump(bi_gram, './Embeddings/bi_gram_bow_alldata.h5')
    joblib.dump(tri_gram, './Embeddings/tri_gram_bow_alldata.h5')

    return one_gram, bi_gram, tri_gram


def create_cosine_distance_matrix(matrix):
    dist = 1 - cosine_similarity(matrix)
    return dist


def run_lda_model(data, num_topics=15, passes=1, max_features=2000):
    """Create LDA model and run it on input data frame.

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame

    Keyword Arguments:
        num_topics {int} -- Number of topics for the LDA model to search for (default: {15})
        passes {int} -- Number of passes over the corpus of document (default: {1})
        max_features {int} -- maximum size of the vocabulary input into the model (default: {2000})

    Returns:
        [LDA Model], [list], [Numpy Array]  -- [description]
    """
    # Load tf features for LDA
    tf_vectorizer = CountVectorizer(
        max_features=max_features, tokenizer=lambda x: word_tokenize(x), max_df=0.8, min_df=0.05)
    tf = tf_vectorizer.fit_transform(data)
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=passes,
                                    learning_method='online', random_state=0,
                                    learning_offset=passes // 2)
    transformed_data = lda.fit_transform(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return lda, tf_feature_names, transformed_data


def get_top_words_lda(model, feature_names, n):
    """Return n most frequent words of the lda model

    Arguments:
        model {LDA Model} -- LDA model of which the top words want to be found
        feature_names {list} -- list of CountVectorizer's features, i.e its indices to words mapping
        n {int} -- number of words to return 

    Returns:
        [type] -- [description]
    """
    topic_top_words = {}
    for topic_index, topic in enumerate(model.components_):
        message = "Topic #%d:" % topic_index
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n - 1:-1]])
        print(message)
        topic_top_words[topic_index] = [feature_names[i]
                                        for i in topic.argsort()[:-n - 1:-1]]
    return topic_top_words


def run_hac_model(data, n_clusters=15, affinity='euclidean'):
    """Simple hierachical agglomerative clustering model to cluster data
    into topic clusters

    Arguments:
        data {Pandas DataFrame} -- Input DataFrames

    Keyword Arguments:
        n_clusters {int} -- Number of topic clusters to assign the data to (default: {15})
        affinity {str} -- Distance Metric (default: {'euclidean'})

    Returns:
        [Numpy Array] -- Array of cluster assignments
    """
    if affinity == 'euclidean':
        hac = AgglomerativeClustering(n_clusters, linkage='ward')
    elif affinity == 'cosine':
        hac = AgglomerativeClustering(
            n_clusters, affinity='cosine', linkage='average')

    labels = hac.fit_predict(data)
    return labels


def word_count_dict(data: pd.DataFrame, text_type='stemmed_text'):
    """Creates a dictionary of the frequency of all the words tokens in 
    the given column, text_type, of the input data.

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame

    Keyword Arguments:
        text_type {str} -- Text column to choose as input into the dictionary (default: {'stemmed_text'})

    Returns:
        [Pandas DataFrame], [dict] -- Returns input DataFrame (unchanged), dictionary of word tokens and their frequency counts
    """
    word_counts = {}
    idx = 0
    data[text_type +
         '_tokens'] = data[text_type].apply(lambda x: word_tokenize(x))
    for index, value in data[text_type + '_tokens'].iteritems():
        idx += 1
        if idx % 1000 == 0:
            print(
                f"Finished: {np.round((idx / len(data[text_type + '_tokens'])) * 100, 2)}% of word_count")
        for word in value:
            word_counts[word] = word_counts.get(word, 0) + 1
    return data, word_counts


def bigram_count_dict(data: pd.DataFrame, text_type='stemmed_text'):
    """Create dictionday of the frequency of all bigram tokens in the given
    column, text_type, of the input data.

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame

    Keyword Arguments:
        text_type {str} -- Text column to choose as input into the dictionary (default: {'stemmed_text'})

    Returns:
        [Pandas DataFrame], [dict] -- Returns input DataFrame (unchanged), dictionary of bigram tokens and their frequency counts.
    """
    bigram_counts = {}
    idx = 0
    data['bigrams'] = data[text_type +
                           '_tokens'].apply(lambda x: [bigram for bigram in ngrams(x, 2)])
    for index, value in data['bigrams'].iteritems():
        idx += 1
        if idx % 1000 == 0:
            print(
                f"Finished : {np.round((idx / len(data[text_type + '_tokens'])) * 100, 2)} % of bigram_count")
        for bigram in value:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    return data, bigram_counts


def word_count_of_corpus(data: pd.DataFrame, text_type='stemmed_text'):
    """Returns the total word count of the document corpus, includes duplicates

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame

    Keyword Arguments:
        text_type {str} -- [Text column to choose as input into the dictionary (default: {'stemmed_text'})

    Returns:
        [int] -- Total word count of the document corpus
    """
    total_word_count = np.sum(data[text_type].apply(lambda x: len(x)))
    return total_word_count


def pointwise_mutual_information(word1: str, word2: str, bigram: tuple,
                                 word_count_dict: dict, bigram_count_dict: dict,
                                 total_word_count: int):
    """Calculates the Pointwise Mutual Information of a given topic cluster, as laid out by 
    'Altuncu, M. T., Mayer, E., Yaliraki, S. N., & Barahona, M. (2019, December). 
     From free text to clusters of content in health records: an unsupervised graph 
     partitioning approach. Applied Network Science , 4 (1)'

    Arguments:
        word1 {str} -- First word
        word2 {str} -- Second word
        bigram {tuple} -- Bigram of first and second word
        word_count_dict {dict} -- Dictionary of word counts for corpus
        bigram_count_dict {dict} -- Dictioncary of bigram counts for corpus
        total_word_count {int} -- Total word count of coprus

    Returns:
        [int] -- PMI score of First and Second word pair
    """
    try:
        pw1 = word_count_dict[word1] / total_word_count
    except:
        pw1 = 0
    try:
        pw2 = word_count_dict[word2] / total_word_count
    except:
        pw2 = 0
    try:
        pbigram = bigram_count_dict[bigram] / total_word_count
    except:
        pbigram = 0
    try:
        pmi = np.log(pbigram / (pw1 * pw2))
    except ValueError:
        pmi = 0

    return pmi


def get_most_common_words(data: pd.DataFrame, cluster: int, max_df=1.0, num_words=15, text_type='stemmed_text',
                          stop_words=None, ngram_range=(1, 1)):
    """Returns most common words of a given topic cluster, for use in calculating the PMI score

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        cluster {int} -- Cluster label

    Keyword Arguments:
        max_df {float} -- Maximum number or proportion of documents that term can occur in to be included in bag of words matrix (default: {1.0})
        num_words {int} -- Number of common words to return (default: {15})
        text_type {str} -- Text column to choose as input into the dictionary (default: {'stemmed_text'})
        stop_words {List} -- List of stop words to exclude from bag of words matrix [description] (default: {None})
        ngram_range {tuple} -- Range of n values of the n-grams to extract from the corpus (default: {(1, 1)})

    Returns:
        [List] -- Highest occuring words
    """
    data = data[data['cluster'] == cluster]
    if len(data) < 1:
        return None
    cv = CountVectorizer(tokenizer=lambda x: word_tokenize(x),
                         max_df=max_df,
                         max_features=num_words,
                         ngram_range=ngram_range,
                         stop_words=stop_words, min_df=1)
    tf_vect = cv.fit_transform(data[text_type])
    sum_words = tf_vect.sum(axis=0)
    word_freq = [(word, sum_words[0, idx])
                 for word, idx in cv.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    word_freq = [word[0] for word in word_freq]
    return word_freq


def get_median_pmi(ordered_word_counts: list, wcd: dict, bgcd: dict, total_word_count: int):
    """Return the Median PMI scores of all the most common words in a given cluster

    Arguments:
        ordered_word_counts {list} -- Most common words, sorted by frequency
        wcd {dict} -- Word count dictionary of corpus
        bgcd {dict} -- Bigram count dictionary of corpus
        total_word_count {int} -- Total word count of corpus

    Returns:
        [int] -- median pmi score of cluster
    """
    pmi_scores = np.zeros(
        (((len(ordered_word_counts) * (len(ordered_word_counts) + 1)) // 2), 1))
    x = 0
    for idx, word1 in enumerate(ordered_word_counts):
        for word2 in ordered_word_counts[idx:]:
            if word1 == word2:
                pass
            else:
                pmi_scores[x] = pointwise_mutual_information(word1, word2, (word1, word2), wcd,
                                                             bgcd,
                                                             total_word_count=total_word_count)
            x += 1
    print('x is equal to:', x)
    return np.median(pmi_scores)


def evaluate_labels(data, labels, num_clusters=15):
    """Returns the Davies Bouldin Index and the Silhouette score of the cluster
    assignments 

    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        labels {Numpy Array} -- Array of cluster assignments

    Keyword Arguments:
        num_clusters {int} -- Number of cluster assingment labels, used only in a print statement for clarity (default: {15})

    Returns:
        [int],[int] -- Davies Bouldin Index, Silhouette Score
    """
    davies_score = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels, random_state=0)
    print(f"Davies Bouldin Score for HAC with {num_clusters}:", davies_score)
    print(f"Silhouette Score for HAC with {num_clusters}:", silhouette)
    return davies_score, silhouette


def run_lda_topic_detection(data, num_topics, wcd, bigram_cd, total_word_count, save_dir, text_type='stemmed_text', passes=100, max_features=10000, num_words=10):
    """Runs the whole topic detection experiment with the LDA model
    Returns the aggregate PMI scores of each topic partition (5, 10, 15)


    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_topics {int} -- Number of topics for the LDA model to detect
        wcd {dict} -- Word count dictionary of corpus
        bigram_cd {dict} -- Bigram count dictionary of corpus
        total_word_count {int} -- Total count of words in corpus
        save_dir {str} -- Directory to save DataFrame with topic cluster assignments

    Keyword Arguments:
        text_type {str} --  (default: {'stemmed_text'})
        passes {int} --  (default: {100})
        max_features {int} -- [description] (default: {10000})
        num_words {int} -- [description] (default: {10})
    """
    print("Running LDA Model")
    lda, tf_feature_names, lda_transformed_data = run_lda_model(data[text_type], num_topics=num_topics, passes=passes,
                                                                max_features=10000)
    lda_top_words = get_top_words_lda(lda, tf_feature_names, num_words)
    lda_topic_data = data.copy(deep='true')
    lda_topic_data['cluster'] = np.argmax(lda_transformed_data, axis=1)
    joblib.dump(
        lda_topic_data, f'./{save_dir}/TopicClustering/lda_topic_data_{num_topics}.h5')
    lda_median_pmi = {}
    lda_median_npmi = {}
    lda_agg_pmi_score = 0
    lda_agg_npmi_score = 0
    for topic_index, topic in enumerate(lda.components_):
        lda_median_pmi[topic_index] = (get_median_pmi(lda_top_words[topic_index], wcd, bigram_cd, total_word_count),
                                       len(lda_topic_data[lda_topic_data['cluster'] == topic_index]))

    for topic, values in lda_median_pmi.items():
        if values[0] == float('-inf'):
            lda_agg_pmi_score += 0

        else:
            lda_agg_pmi_score += (values[0] *
                                  (values[1] / len(lda_topic_data)))

    print("LDA topics:", get_top_words_lda(lda, tf_feature_names, 15))
    print("LDA PMI Score: ", lda_agg_pmi_score)

    return lda_agg_pmi_score

def get_agg_pmi_score(data, label_set,name,num_topics,save_dir,num_words,wcd,bigram_cd,total_word_count,text_type='stemmed_text'):
    """Returns the weighted mean pmi score of the median pmi scores of each cluster, given a cluster assignment (label set)
    
    Arguments:
        data {Pandas DataFrame} -- Input Data
        label_set {Numpy Array} -- Array of cluster assignments
        name {str} -- Type of input data i.e. TF-IDF matrix/Bag of words matrix
        num_topics {int} -- Number of topics clusters
        save_dir {str} -- Directory to save labeled dataframe to
        num_words {int} -- Number of words to find pmi score of
        wcd {dict} -- Word count dictionary of corpus
        bigram_cd {dict} -- Bigram count dictionary of corpus
        total_word_count {int} -- Total count of words in corpus
    
    Keyword Arguments:
        text_type {str} -- Text column to choose as input into the dictionary (default: {'stemmed_text'})

    Returns:
        [int] -- weighted mean pmi score of label set
    """
    agg_pmi_score = 0
    agg_pmi_dict = {}
    median_pmi_scores = {}

    data['cluster'] = label_set
    joblib.dump(
        data, f'./{save_dir}/TopicClustering/{name}_topic_data_{num_topics}.h5')
    for label in np.unique(label_set):
        common_words = get_most_common_words(
            data, label, num_words=num_words, text_type=text_type)
        if common_words is None:
            pass
        median_pmi_scores[label] = (get_median_pmi(list(common_words), wcd, bigram_cd, total_word_count),
                                    len(data[data['cluster'] == label]))

    for cluster, values in median_pmi_scores.items():
        if values[0] == float('-inf'):
            agg_pmi_score += 0
        else:
            agg_pmi_score += (values[0] * (values[1] / len(data)))

    agg_pmi_dict[name] = agg_pmi_score
    print(agg_pmi_dict)

    return agg_pmi_dict


def run_doc2vec_hac_experiment(data, num_topics, save_dir):
    """Runs a Hierachical Agglomerative Clustering model on the doc2vec embeddings of the 
    data
    
    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_topics {int} -- Number of topic clusters to assign data to
        save_dir {str} -- Directory to save labels to 
    
    Returns:
        [Numpy Array], [int], [int] -- Array of cluster assignments, Davies Bouldin Index, Silhoeutte Score 
    """
    print("Loading Doc2vec Model")
    doc2vec = Doc2Vec.load(
        './SavedModels/saved_doc2vec_eval_model_clustering')
    embeddings = get_embeddings(data, doc2vec)
    print("Running HAC")
    embeddings_labels = run_hac_model(
        embeddings, n_clusters=num_topics, affinity='euclidean')
    print("Evaluating")
    embedding_db, embedding_sl = evaluate_labels(
        embeddings, embeddings_labels, num_clusters=num_topics)
    joblib.dump(embeddings_labels, save_dir +
                f'embeddings_label_{num_topics}.h5')
    return embeddings_labels, embedding_db, embedding_sl


def run_tf_idf_hac_experiment(data, num_topics, save_dir,text_type = 'stemmed_text', force=False):
    """Runs a Hierachical Agglomerative Clustering model on the tf-idf matrices of the 
    data
    
    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_topics {int} -- Number of topic clusters to assign data to
        save_dir {str} -- Directory to save labels to 
    
    Keyword Arguments:
        text_type {str} -- Text column to choose as input into the dictionary (default: {'stemmed_text'})
    Returns:
        [Numpy Array], [int], [int] -- Array of cluster assignments, Davies Bouldin Index, Silhoeutte Score 
    """
    if force == False:
        if os.path.isfile(f'./{save_dir}/one_gram_labels_{num_topics}.h5'):
            one_gram_labels = joblib.load(
                f'./{save_dir}/one_gram_labels_{num_topics}.h5')
    else:
        print("Testing all data tf-idf vectors")
        one_gram, bi_gram, tri_gram = create_tf_idf(
            data, 2000, text_type=text_type, force=force)

        print('Testing one-gram')
        one_gram_labels = run_hac_model(
            one_gram.todense(), n_clusters=num_topics)
        one_gram_db, one_gram_sl = evaluate_labels(
            one_gram.todense(), one_gram_labels, num_clusters=num_topics)
        joblib.dump(one_gram_labels,
                    f'./{save_dir}/one_gram_labels_{num_topics}.h5')

    return one_gram_labels, one_gram_db, one_gram_sl


def run_bow_hac_experiment(data, num_topics,save_dir, text_type = 'stemmed_text', force=False):
    """Runs a Hierachical Agglomerative Clustering model on the Bag of Words matrices of the 
    data
    
    Arguments:
        data {Pandas DataFrame} -- Input DataFrame
        num_topics {int} -- Number of topic clusters to assign data to
        save_dir {str} -- Directory to save labels to 
    Returns:
        [Numpy Array], [int], [int] -- Array of cluster assignments, Davies Bouldin Index, Silhoeutte Score 
    """
    if force == False:
        if os.path.isfile(f'./{save_dir}/one_gram_labels_{num_topics}.h5'):
            one_gram_labels = joblib.load(
                f'./{save_dir}/one_gram_labels_{num_topics}.h5')
    else:
        print("Testing all data BOW vectors")
        one_gram_bow, bi_gram_bow, tri_gram_bow = create_bow_representations(data, 2000, text_type=text_type,
                                                                             force=force)
        print('Testing one-gram bow')
        one_gram_bow_labels = run_hac_model(
            one_gram_bow.todense(), n_clusters=num_topics)
        one_gram_bow_db, one_gram_bow_sl = evaluate_labels(one_gram_bow.todense(), one_gram_bow_labels,
                                                           num_clusters=num_topics)
        joblib.dump(one_gram_bow_labels,
                    f'./{save_dir}/one_gram_bow_labels_{num_topics}.h5')
    return one_gram_bow_labels, one_gram_bow_db, one_gram_bow_sl


def main(force=False, dataset='all'):
    print("loading Data")
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
        politifact = joblib.load()
        data = pd.DataFrame()
        for df in [politifact]:
            data = data.append(df)

    # Create results folder if not already created
    if not os.path.exists('results'):
        os.makedirs('results')
    save_dir = f'./results/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Run topic experiment
    for num in [5, 10, 15]:
        num_topics = num
        num_words = 10
        text_type = 'stemmed_text'

        # Embeddings experiment
        embedding_labels, embedding_db, embedding_sl = run_doc2vec_hac_experiment(
            data, num_topics=num_topics, save_dir=save_dir)

        # TF-IDF Experiment
        one_gram_labels, one_gram_db, one_gram_sl = run_tf_idf_hac_experiment(
            data, text_type=text_type, num_topics=num_topics, dataset=dataset, save_dir=save_dir, force=force)

        # BOW Experiments
        one_gram_bow_labels, one_gram_bow_db, one_gram_bow_sl = run_bow_hac_experiment(
            data, text_type=text_type, num_topics=num_topics, dataset=dataset, save_dir=save_dir, force=force)

        # Create dictionaries of results
        labels = {'one_gram_labels': one_gram_labels,
                  'embeddings_labels': embedding_labels, 'one_gram_bow_labels': one_gram_bow_labels}
        scores = {'embeddings': (embedding_db, embedding_sl),
                  'one_gram_tfidf': (one_gram_db, one_gram_sl),
                  'one_gram_bow': (one_gram_bow_db, one_gram_bow_sl)}

        data, wcd = word_count_dict(data, text_type)
        data, bigram_cd = bigram_count_dict(data, text_type)
        total_word_count = word_count_of_corpus(data, text_type)
        label_idx = 0
        
        if not os.path.exists(save_dir+'/TopicClustering'):
            os.makedirs(save_dir+"/TopicClustering")

        # LDA PMI
        lda_agg_pmi_score = run_lda_topic_detection(data=data, num_topics=num_topics, wcd=wcd,
                                                bigram_cd=bigram_cd, total_word_count=total_word_count, save_dir=save_dir)
        # Clustering PMI
        for name, label_set in labels.items():
            agg_pmi_dict = get_agg_pmi_score(data,label_set=label_set,name=name,num_topics=num_topics,save_dir=save_dir,
                num_words=num_words,text_type=text_type,wcd=wcd,bigram_cd=bigram_cd,total_word_count=total_word_count)

        # Write metrics to csv
        if not os.path.exists(f'{save_dir}/TopicClustering/CSV/'):
            os.makedirs(f'{save_dir}/TopicClustering/CSV/')
        score_results_file = open(
            f'{save_dir}/TopicClustering/CSV/clustering_scores.csv', 'a')
        logwriter = csv.DictWriter(score_results_file,
                                   fieldnames=['method', 'number_of_topics', 'Davies_Bouldin_Score',
                                               'Silhouette_Score'])
        logwriter.writeheader()
        for key, value in scores.items():
            logwriter.writerow(
                dict(method=key, number_of_topics=num_topics, Davies_Bouldin_Score=value[0],
                     Silhouette_Score=value[1]))

        # Write PMI to csv
        pmi_results_file = open(
            f'{save_dir}/TopicClustering/CSV/pmi_results.csv', 'a')
        logwriter = csv.DictWriter(pmi_results_file, fieldnames=[
            'method', 'number_of_topics', 'pmi'])
        logwriter.writeheader()
        for name, pmi_score in agg_pmi_dict.items():
            logwriter.writerow(
                dict(method=name, number_of_topics=num_topics, pmi=pmi_score))
        logwriter.writerow(
            dict(method='lda', number_of_topics=num_topics, pmi=lda_agg_pmi_score))

if __name__ == '__main__':
    main(force=True, dataset='politifact')
