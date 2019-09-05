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
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from gensim.models import Doc2Vec
from CreateEmbeddings import create_tagged_documents
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from nltk import ngrams
import os
import csv
import pyreadr
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.phrases import npmi_scorer


def get_representation(representation='tf-idf', data='gossip_cop'):
    representation = joblib.load(
        './Embeddings/' + representation + data + '_clustering.h5')
    return representation


def get_embeddings(data, model):
    print("Creating Tagged Docs")
    training_data = create_tagged_documents(data)
    print("Infering Vectors")
    embeddings = np.array([model.infer_vector(doc.words, epochs=50, alpha=0.01, min_alpha=0.0001)
                           for doc in training_data])
    return embeddings


def create_tf_idf(data, num_words, word_type='text', force=False):
    if force == False:
        if os.path.isfile("./Embeddings/one_gram_tfidf_alldata.h5"):
            one_gram = joblib.load('./Embeddings/one_gram_tfidf_alldata.h5')
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
        print("Creating one-gram vector")
        one_gram = tf_vect.fit_transform(data[word_type])
        print("Creating bi-gram vector")
        bi_gram = bigram_tf_vect.fit_transform(data[word_type])
        print("Creating tri-gram vector")
        tri_gram = trigram_tf_vect.fit_transform(data[word_type])
        joblib.dump(one_gram, './Embeddings/one_gram_tfidf_alldata.h5')
        joblib.dump(bi_gram, './Embeddings/bi_gram_tfidf_alldata.h5')
        joblib.dump(tri_gram, './Embeddings/tri_gram_tfidf_alldata.h5')

    return one_gram, bi_gram, tri_gram


def create_bow_representations(data, num_words, word_type='text', force=False):
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
    print("Creating one-gram BOW vector")
    one_gram = one_gram_bow.fit_transform(data[word_type])
    print("Creating bi-gram BOW vector")
    bi_gram = bi_gram_bow.fit_transform(data[word_type])
    print('Creating tri-gram BOW vector')
    tri_gram = tri_gram_bow.fit_transform(data[word_type])
    joblib.dump(one_gram, './Embeddings/one_gram_bow_alldata.h5')
    joblib.dump(bi_gram, './Embeddings/bi_gram_bow_alldata.h5')
    joblib.dump(tri_gram, './Embeddings/tri_gram_bow_alldata.h5')
    return one_gram, bi_gram, tri_gram


def create_cosine_distance_matrix(matrix):
    dist = 1 - cosine_similarity(matrix)
    return dist


def run_lda(data, num_topics=15, update_every=1, passes=1, max_features=2000):
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


def get_top_words_lda(model, feature_names, n_top_words):
    topic_top_words = {}
    for topic_index, topic in enumerate(model.components_):
        message = "Topic #%d:" % topic_index
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        topic_top_words[topic_index] = [feature_names[i]
                                        for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topic_top_words


def run_hac(data, n_clusters=15, affinity='euclidean'):
    if affinity == 'euclidean':
        hac = AgglomerativeClustering(n_clusters, linkage='ward')
    elif affinity == 'cosine':
        hac = AgglomerativeClustering(
            n_clusters, affinity='cosine', linkage='average')

    labels = hac.fit_predict(data)
    return labels


def run_kmeans(data, n_clusters=15):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)
    return labels


def word_count_dict(data: pd.DataFrame, text_type='stemmed_text'):
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
    total_word_count = np.sum(data[text_type].apply(lambda x: len(x)))
    return total_word_count


def pointwise_mutual_information(word1: str, word2: str, bigram: tuple,
                                 word_count_dict: dict, bigram_count_dict: dict,
                                 total_word_count: int):
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


def get_normalized_npmi(word1: str, word2: str, bigram: tuple,
                        word_count_dict: dict, bigram_count_dict: dict,
                        total_word_count: int):
    try:
        npmi = npmi_scorer(word_count_dict[word1],
                           word_count_dict[word2], bigram_count_dict[bigram],
                           len_vocab=0, min_count=1, corpus_word_count=total_word_count)
    except KeyError:
        npmi = 0
    return npmi


def create_dict_per_row(row):
    d = {}
    for word in row:
        d[word] = d.get(word, 0) + 1
    return d


def get_most_common_words(data: pd.DataFrame, cluster: int, max_df=1.0, num_words=15, text_type='stemmed_text',
                          stop_words=None, cluster_col='cluster', ngram_range=(1, 1)):
    data = data[data[cluster_col] == cluster]
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


def get_median_npmi(ordered_word_counts: list, wcd: dict, bgcd: dict, total_word_count: int):
    npmi_scores = np.zeros(
        (((len(ordered_word_counts) * (len(ordered_word_counts) + 1)) // 2), 1))
    x = 0
    for idx, word1 in enumerate(ordered_word_counts):
        for word2 in ordered_word_counts[idx:]:
            if word1 == word2:
                pass
            else:
                npmi_scores[x] = get_normalized_npmi(word1, word2, (word1, word2), wcd,
                                                     bgcd,
                                                     total_word_count=total_word_count)
            x += 1
    return np.median(npmi_scores)


def evaluate_labels(data, labels, num_clusters=15):
    davies_score = davies_bouldin_score(data, labels)
    silhouette = silhouette_score(data, labels, random_state=0)
    print(f"Davies Bouldin Score for HAC with {num_clusters}:", davies_score)
    print(f"Silhouette Score for HAC with {num_clusters}:", silhouette)
    return davies_score, silhouette


def main(force=False, dataset='all'):
    print("loading Data")
    if dataset == 'all':
        politifact = joblib.load(
            './Data/Preprocessed/politifact_clustering_large.h5')
        gossipcop = joblib.load(
            './Data/Preprocessed/gossipcop_clustering_large.h5')
        """
        politifact_small = joblib.load(
            './Data/Preprocessed/PolitiFact_clustering_small.h5')
        buzzfeed = joblib.load(
            './Data/Preprocessed/BuzzFeed_clustering_small.h5')
        """
        data = pd.DataFrame()
        for df in [politifact, gossipcop]:  # politifact_small, buzzfeed,
            data = data.append(df)
    elif dataset == 'gossipcop':
        gossipcop = joblib.load(
            './Data/Preprocessed/gossipcop_clustering_large.h5')
        data = gossipcop

    else:
        politifact = joblib.load(
            './Data/Preprocessed/politifact_clustering_large.h5')
        """
        politifact_small = joblib.load(
            './Data/Preprocessed/PolitiFact_clustering_small.h5')
        buzzfeed = joblib.load(
            './Data/Preprocessed/BuzzFeed_clustering_small.h5')
        """
        data = pd.DataFrame()
        for df in [politifact]:  # politifact_small, buzzfeed
            data = data.append(df)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_dir = f'./results/{dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for num in [5, 10, 15]:
        num_topics = num
        num_words = 10
        word_type = 'stemmed_text'
        print("Running LDA Model")
        lda, tf_feature_names, lda_transformed_data = run_lda(data[word_type], num_topics=num_topics, passes=100,
                                                              max_features=10000)
        lda_top_words = get_top_words_lda(lda, tf_feature_names, num_words)

        print("Loading Doc2vec Model")
        doc2vec = Doc2Vec.load(
            './SavedModels/saved_doc2vec_eval_model_clustering')
        embeddings = get_embeddings(data, doc2vec)
        print("Running HAC")
        embeddings_labels = run_hac(
            embeddings, n_clusters=num_topics, affinity='euclidean')
        print("Evaluating")
        embedding_db, embedding_sl = evaluate_labels(
            embeddings, embeddings_labels, num_clusters=num_topics)
        joblib.dump(embeddings_labels, save_dir +
                    f'embeddings_label_{num_topics}.h5')

        if force == False:
            if os.path.isfile(f'./results/{dataset}/one_gram_labels_{num_topics}.h5'):
                one_gram_labels = joblib.load(
                    f'./results/{dataset}/one_gram_labels_{num_topics}.h5')
            if os.path.isfile(f'./results/{dataset}/bi_gram_labels_{num_topics}.h5'):
                bi_gram_labels = joblib.load(
                    f'./results/{dataset}/bi_gram_labels_{num_topics}.h5')
            if os.path.isfile(f'./results/{dataset}/tri_gram_labels_{num_topics}.h5'):
                tri_gram_labels = joblib.load(
                    f'./results/{dataset}/tri_gram_labels_{num_topics}.h5')
            if os.path.isfile(f'./results/{dataset}/mixed_gram_labels_{num_topics}.h5'):
                mixed_gram_labels = joblib.load(
                    f'./results/{dataset}/mixed_gram_labels_{num_topics}.h5')
        else:
            print("Testing all data tf-idf vectors")
            one_gram, bi_gram, tri_gram = create_tf_idf(
                data, 2000, word_type=word_type, force=force)

            print('Testing one-gram')
            one_gram_labels = run_hac(
                one_gram.todense(), n_clusters=num_topics)
            one_gram_db, one_gram_sl = evaluate_labels(
                one_gram.todense(), one_gram_labels, num_clusters=num_topics)
            joblib.dump(one_gram_labels,
                        f'./{save_dir}/one_gram_labels_{num_topics}.h5')
            """
            print('Testing one-gram cosine matrix')
            og_cosine = create_cosine_distance_matrix(one_gram)
            og_cosine_labels = run_hac(og_cosine, n_clusters=num_topics)
            og_cosine_db, og_cosine_sl = evaluate_labels(
                og_cosine, og_cosine_labels, num_clusters=num_topics)
            joblib.dump(og_cosine_labels,
                        f'./results/one_gram_cosine_labels_{num_topics}.h5')
            
            print("Testing bi-gram")
            bi_gram_labels = run_hac(bi_gram.todense(), n_clusters=num_topics)
            bi_gram_db, bi_gram_sl = evaluate_labels(
                bi_gram.todense(), bi_gram_labels, num_clusters=num_topics)
            joblib.dump(bi_gram_labels,
                        f'./{save_dir}/bi_gram_labels_{num_topics}.h5')
            
            print('Testing bi-gram cosine matrix')
            bg_cosine = create_cosine_distance_matrix(bi_gram)
            bg_cosine_labels = run_hac(bg_cosine, n_clusters=num_topics)
            bg_cosine_db, bg_cosine_sl = evaluate_labels(
                bg_cosine, bg_cosine_labels, num_clusters=num_topics)
            joblib.dump(bg_cosine_labels,
                        f'./results/bi_gram_cosine_labels_{num_topics}.h5')
        
            print("Testing Tri-gram")
            tri_gram_labels = run_hac(
                tri_gram.todense(), n_clusters=num_topics)
            tri_gram_db, tri_gram_sl = evaluate_labels(
                tri_gram.todense(), tri_gram_labels, num_clusters=num_topics)
            joblib.dump(tri_gram_labels,
                        f'./{save_dir}/tri_gram_labels_{num_topics}.h5')
            
            print('Testing tri-gram cosine matrix')
            tg_cosine = create_cosine_distance_matrix(tri_gram)
            tg_cosine_labels = run_hac(tg_cosine, n_clusters=num_topics)
            tg_cosine_db, tg_cosine_sl = evaluate_labels(
                tg_cosine, tg_cosine_labels, num_clusters=num_topics)
            joblib.dump(tg_cosine_labels,
                        f'./results/tri_gram_cosine_labels_{num_topics}.h5')
            """
            print("Testing all data BOW vectors")
            one_gram_bow, bi_gram_bow, tri_gram_bow = create_bow_representations(data, 2000, word_type=word_type,
                                                                                 force=force)
            print('Testing one-gram bow')
            one_gram_bow_labels = run_hac(
                one_gram_bow.todense(), n_clusters=num_topics)
            one_gram_bow_db, one_gram_bow_sl = evaluate_labels(one_gram_bow.todense(), one_gram_bow_labels,
                                                               num_clusters=num_topics)
            joblib.dump(one_gram_bow_labels,
                        f'./{save_dir}/one_gram_bow_labels_{num_topics}.h5')
            """
            print('Testing one-gram bow cosine matrix')
            ogb_cosine = create_cosine_distance_matrix(one_gram_bow)
            ogb_cosine_labels = run_hac(ogb_cosine, n_clusters=num_topics)
            ogb_cosine_db, ogb_cosine_sl = evaluate_labels(
                ogb_cosine, ogb_cosine_labels, num_clusters=num_topics)
            joblib.dump(ogb_cosine_labels,
                        f'./results/one_gram_bow_cosine_labels_{num_topics}.h5')

            print("Testing bi-gram bow")
            bi_gram_bow_labels = run_hac(
                bi_gram_bow.todense(), n_clusters=num_topics)
            bi_gram_bow_db, bi_gram_bow_sl = evaluate_labels(bi_gram_bow.todense(), bi_gram_bow_labels,
                                                             num_clusters=num_topics)
            joblib.dump(bi_gram_bow_labels,
                        f'./{save_dir}/bi_gram_bow_labels_{num_topics}.h5')

            print('Testing bi-gram bow cosine matrix')
            bgb_cosine = create_cosine_distance_matrix(bi_gram_bow)
            bgb_cosine_labels = run_hac(bgb_cosine, n_clusters=num_topics)
            bgb_cosine_db, bgb_cosine_sl = evaluate_labels(
                bgb_cosine, bgb_cosine_labels, num_clusters=num_topics)
            joblib.dump(bgb_cosine_labels,
                        f'./results/bi_gram_bow_cosine_labels_{num_topics}.h5')

            print("Testing Tri-gram bow")
            tri_gram_bow_labels = run_hac(
                tri_gram_bow.todense(), n_clusters=num_topics)
            tri_gram_bow_db, tri_gram_bow_sl = evaluate_labels(tri_gram_bow.todense(), tri_gram_bow_labels,
                                                               num_clusters=num_topics)
            joblib.dump(tri_gram_bow_labels,
                        f'./{save_dir}/tri_gram_bow_labels_{num_topics}.h5')

            print('Testing Tri-gram bow Cosine Matrix')
            tgb_cosine = create_cosine_distance_matrix(tri_gram_bow)
            tgb_cosine_labels = run_hac(tgb_cosine, n_clusters=num_topics)
            tgb_cosine_db, tgb_cosine_sl = evaluate_labels(
                tgb_cosine, tgb_cosine_labels, num_clusters=num_topics)
            joblib.dump(tgb_cosine_labels,
                        f'./results/tri_gram_bow_cosine_labels_{num_topics}.h5')
            """
        labels = {'one_gram_labels': one_gram_labels,
                  'embeddings_labels': embeddings_labels, 'one_gram_bow_labels': one_gram_bow_labels}
        """
        'one_gram_cosine_labels': og_cosine_labels
        'one_gram_bow_cosine_labels': ogb_cosine_labels
          'bi_gram_labels': bi_gram_labels, 'tri_gram_labels': tri_gram_labels,®¥
          'bi_gram_bow_labels': bi_gram_bow_labels,
                  'tri_gram_bow_labels': tri_gram_bow_labels,
                  }
        'bi_gram_cosine_labels': bg_cosine_labels,
        'tri_gram_cosine_labels': tg_cosine_labels, 
        'bi_gram_bow_cosine_labels': bgb_cosine_labels, 'tri_gram_bow_cosine_labels': tgb_cosine_labels}
        """
        scores = {'embeddings': (embedding_db, embedding_sl),
                  'one_gram_tfidf': (one_gram_db, one_gram_sl),
                  'one_gram_bow': (one_gram_bow_db, one_gram_bow_sl)
                  }
        """
        'one_gram_tfidf_cosine': (og_cosine_db, og_cosine_sl),
        'one_gram_bow_cosine': (ogb_cosine_db, ogb_cosine_sl)
        ,
                  'bi_gram_tfidf': (bi_gram_db, bi_gram_sl),
                  'tri_gram_tfidf': (tri_gram_db, tri_gram_sl),
                  'bi_gram_bow': (bi_gram_bow_db, bi_gram_bow_sl),
                  'tri_gram_bow': (tri_gram_bow_db, tri_gram_bow_sl),
        
        'bi_gram_tfidf_cosine': (bg_cosine_db, bg_cosine_sl),
        'tri_gram_tfidf_cosine': (tg_cosine_db, tg_cosine_sl),
        'bi_gram_bow_cosine': (bgb_cosine_db, bgb_cosine_sl),
        'tri_gram_bow_cosine': (tgb_cosine_db, tgb_cosine_sl)}
        """

        data, wcd = word_count_dict(data, word_type)
        data, bigram_cd = bigram_count_dict(data, word_type)

        total_word_count = word_count_of_corpus(data, word_type)
        median_pmi_scores = {}
        median_npmi_scores = {}
        agg_pmi_dict = {}
        agg_npmi_dict = {}
        label_idx = 0
        if not os.path.exists(save_dir+'/TopicClustering'):
            os.makedirs(save_dir+"/TopicClustering")
        # LDA PMI
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
            lda_median_npmi[topic_index] = (
                get_median_npmi(
                    lda_top_words[topic_index], wcd, bigram_cd, total_word_count),
                len(lda_topic_data[lda_topic_data['cluster'] == topic_index]))

        for topic, values in lda_median_pmi.items():
            if values[0] == float('-inf'):
                lda_agg_pmi_score += 0

            else:
                lda_agg_pmi_score += (values[0] *
                                      (values[1] / len(lda_topic_data)))
        for topic, values in lda_median_npmi.items():
            if values[0] == float('-inf'):
                lda_agg_pmi_score += 0

            else:
                lda_agg_npmi_score += (values[0] *
                                       (values[1] / len(lda_topic_data)))

        # Clustering PMI
        for name, label_set in labels.items():
            agg_pmi_score = 0
            agg_npmi_score = 0
            data['cluster'] = label_set
            joblib.dump(
                data, f'./{save_dir}/TopicClustering/{name}_topic_data_{num_topics}.h5')
            for label in np.unique(label_set):
                common_words = get_most_common_words(
                    data, label, num_words=num_words, text_type=word_type)
                if common_words is None:
                    pass
                median_pmi_scores[label] = (get_median_pmi(list(common_words), wcd, bigram_cd, total_word_count),
                                            len(data[data['cluster'] == label]))
                median_npmi_scores[label] = (get_median_npmi(list(common_words), wcd, bigram_cd, total_word_count),
                                             len(data[data['cluster'] == label]))

            for cluster, values in median_pmi_scores.items():
                if values[0] == float('-inf'):
                    agg_pmi_score += 0
                else:
                    agg_pmi_score += (values[0] * (values[1] / len(data)))

            for cluster, values in median_npmi_scores.items():
                if values[0] == float('-inf'):
                    agg_npmi_score += 0
                else:
                    agg_npmi_score += (values[0] * (values[1] / len(data)))

            agg_pmi_dict[name] = agg_pmi_score
            agg_npmi_dict[name] = agg_npmi_score

        print("LDA topics:", get_top_words_lda(lda, tf_feature_names, 15))
        print("LDA PMI Score: ", lda_agg_pmi_score)
        print("LDA NPMI Score: ", lda_agg_npmi_score)
        print(agg_pmi_dict)
        print(agg_npmi_dict)
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
