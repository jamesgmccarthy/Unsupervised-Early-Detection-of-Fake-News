'''Code adapted from 
 https://github.com/RaRe-Technologies/gensim/blob/ca0dcaa1eca8b1764f6456adac5719309e0d8e6d/docs/notebooks/doc2vec-IMDB.ipynb
'''
import time
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score, confusion_matrix, \
    accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import statsmodels.api as sm
from random import sample
import os

def create_tagged_documents(data, text_type='word_tokens'):
    documents = [TaggedDocument(doc, [i])
                 for i, doc in enumerate(data[text_type])]
    return documents


def create_doc2vec_model():
    eval_model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window_size=15,
                         min_count=20, sample=1e-5, negative=5, epochs=50, workers=4)
    models = {'eval_model': eval_model}
    return models


def train_doc2vec_model(docs, models, dataset):
    for name, model in models.items():
        print(f"Training {name}")
        start = time.time()
        model.build_vocab(docs)
        model.train(docs, total_examples=len(docs), epochs=model.epochs)
        print('Total time taken: ', time.time() - start)
        print(f"Saving {name}")
        model.save('./SavedModels/saved_doc2vec_' + name+"_"+dataset)
        models[name] = model
    return models


def concatenate_models(model1, model2):
    model = ConcatenatedDoc2Vec([model1, model2])
    return model


# TESTING DOC2VECS


def logistic_predictor_from_data(train_targets, train_regressors):
    """Fit a statsmodel logistic predictor on data"""
    logit = MLPClassifier(hidden_layer_sizes=(300, 100, 2), random_state=0)
    predictor = logit.fit(train_regressors, train_targets)
    return predictor


def error_rate_for_model(test_model, train_data, train_labels, test_data, test_labels,
                         reinfer_train=False, reinfer_test=False, infer_steps=None, infer_alpha=None,
                         infer_subsample=0.2):
    if reinfer_train:
        train_regressors = [test_model.infer_vector(
            doc.words, steps=infer_steps, alpha=infer_alpha) for doc in train_data]
    else:
        train_regressors = [test_model.docvecs[doc.tags[0]]
                            for doc in train_data]
    train_labels = list(train_labels)
    predictor = logistic_predictor_from_data(train_labels, train_regressors)
    if reinfer_test:
        test_regressors = [test_model.infer_vector(
            doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]]
                           for doc in test_data]

    # Predict and eval
    test_predictons = predictor.predict(test_regressors)
    f1score = f1_score(test_labels, test_predictons)
    print('f1_score: ', f1score)
    acc = accuracy_score(test_labels, test_predictons)
    print('Accuaracy:', acc)


def main(retrain_model=False, dataset='clustering'):
    assert gensim.models.doc2vec.FAST_VERSION > -1
    print("Loading Datasets")
    # load datasets
    politifact_large = joblib.load(
        f"./Data/Preprocessed/politifact_{dataset}_large.h5")
    buzz_small = joblib.load(f'./Data/Preprocessed/BuzzFeed_fnd_small.h5')
    politifact_small = joblib.load(
        f'./Data/Preprocessed/PolitiFact_fnd_small.h5')
    gossip_cop = joblib.load(
        f"./Data/Preprocessed/gossipcop_{dataset}_large.h5")
    # combine datasets
    data = politifact_large.append(gossip_cop, ignore_index=True)
    data = data.append(buzz_small, ignore_index=True)
    data = data.append(politifact_small, ignore_index=True)
    # Create tagged documents for datasets
    print("Creating Tagged Documents")
    all_tagged_docs = create_tagged_documents(data)
    if not os.path.isdir('./SaveModels'):
        os.makedirs('./SavedModels')
    # Train models on full dataset
    if retrain_model == False:
        eval_model = Doc2Vec.load(
            'f./SavedModels/saved_doc2vec_eval_model_{datset}')
        model = {'eval_model': eval_model}
    else:
        model = create_doc2vec_model()
        model = train_doc2vec_model(all_tagged_docs, model, dataset=dataset)


if __name__ == "__main__":
    main(retrain_model=True)
