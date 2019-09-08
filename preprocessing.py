import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
import spacy
from spacy.matcher import PhraseMatcher
import os


def load_dataset(path):
    return joblib.load(path)


def lower_case_text(data):
    data['text'] = data['text'].apply(
        lambda x: " ".join(y.lower() for y in x.split()))
    return data


def remove_punctuation(data, full_stop=False):
    if full_stop:
        data['text'] = data['text'].apply(
            lambda x: re.sub(r'[^\w\s]', '', x))
    else:
        data['text'] = data['text'].apply(
            lambda x: re.sub(r'[^\w\s.]', '', x))
        data['text'] = data['text'].apply(
            lambda x: re.sub(r'[-]', ' ', x))
        data['text'] = data['text'].apply(
            lambda x: re.sub(r'[_]', '', x)
        )
    return data


def remove_newlines(data):
    data['text'] = data['text'].apply(lambda x: " ".join(x.splitlines()))
    return data


def remove_stop_words(data):
    stop = set(stopwords.words('english'))
    data['text'] = data['text'].apply(
        lambda x: " ".join(y for y in x.split() if y not in stop))
    return data


def remove_digits(data):
    data['text'] = data['text'].apply(lambda x: re.sub('[0-9]', "", x))
    return data


def stem_words(data):
    st = SnowballStemmer(language='english')
    data['stemmed_text'] = data['text'].apply(
        lambda x: " ".join([st.stem(word) for word in x.split()]))
    return data


def lemmatize_word(data):
    tagged_text = nltk.pos_tag(data['text'])
    lemmatizer = WordNetLemmatizer()
    leftovers = set([word for (word, pos) in tagged_text if pos in ['NN', 'NNS', 'NNP', 'NNPs', 'VB',
                                                                    'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                                                                    'JJ', 'JJR', 'JJS']])
    data['lemmatized_text'] = data['text'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split() if word in leftovers]))
    return data


def tokenize_words(data):
    data['word_tokens'] = data['text'].apply(lambda x: word_tokenize(x))
    return data


def tokenize_sentences(data):
    data['sent_tokens'] = data['text'].apply(lambda x: sent_tokenize(x))
    return data


def convert_labels_to_categorical(data):
    data['label'] = data['label'].apply(lambda x: 0 if x == 'real' else 1)
    return data


def remove_cookied_articles(data):
    """"Some artilces didn't download properly so contain
    just the website requesting premission for using cookies
    Arguments:
        data {Pandas DataFrame} -- [description]
    Returns:
        data {Pandas Dataframe}
    """
    data['iscookies'] = data['text'].apply(
        lambda x: True if 'cookies' in x.split(" ") else False)
    text_of_is_cookies = data.loc[data['iscookies'] == True, 'text']
    cookie_recog = {}
    for i in text_of_is_cookies:
        cookie_recog[i] = cookie_recog.get(i, 0) + 1
    non_unique = {key: value for key,
                  value in cookie_recog.items() if value > 2}
    non_unique = list(non_unique.keys())
    data['not_unique'] = data['text'].apply(
        lambda x: True if x in non_unique else False)
    data = data[data.iscookies == False]
    return data


def remove_unavailable_websites(data):
    """Phrases found through exploration, perhaps try to make dynamic
    """

    phrases = ['currently unavailable', 'please use supported',
               'sorry content', 'watch queue',
               'article removed', 'privacy site welcome']
    for phrase in phrases:
        data[phrase] = data['text'].apply(
            lambda x: True if phrase in x else False)
    for phrase in phrases:
        data = data.drop(data[data[phrase] == True].index)
        data = data.drop([phrase], axis=1)
    return data


def clean_url(url):
    temp = re.sub(r'http', '', url)
    temp = temp.lstrip('s/:w')
    temp = temp.lstrip('.')
    parts = temp.split(".")
    try:
        index = parts.index("wikipedia")
        return parts[index]
    except ValueError:
        return parts[0]


def remove_two_labeled(data):
    df = data[['text', 'label']]
    d = {}

    # create dict with text as key, list of labels and indices as value
    for row in df.itertuples():
        d[(row[1])] = d.get((row[1]), []) + [row[2], row[0]]

    # recreate dict including only articles with two labels (real and fake), exclude labels in new value
    d = {key: sorted(list(set(value)))[2:]
         for key, value in d.items() if (0 in value) & (1 in value)}

    drop_array = []
    for values in d.values():
        for value in values:
            drop_array.append(value)
    data = data.drop(index=drop_array)
    return data


def remove_noisey_duplicates(data):
    """"There are number of duplicated articles containing useless text
    eg: raccoons keep calm carry on cute raccoons appear waiting fed front one looking keep every...
    occurs in approx 40 articles
    """
    df = data[['text']]
    d = {}
    # Create dict with text as key, list indices as value
    for row in df.itertuples():
        d[(row[1])] = d.get((row[1]), []) + [row[0]]
    # Exclude values with less than 4 occurences and text lengths less than 500 characters
    # these values were arrived at after testing various different values
    d = {key: set(value)
         for key, value in d.items() if (len(value) > 4) & (len(key) < 500)}
    drop_array = []
    for values in d.values():
        for value in values:
            drop_array.append(value)
    data = data.drop(index=drop_array)
    return data


def remove_dupicated_articles(data):
    """There are a number of legitimate articles that are present more than once in the dataset
    Keep first of these
    """
    data = data.drop_duplicates(subset='text', keep='first')
    return data


def remove_articles_explored(data):
    """
    Remove the articles found to be unnecessary through exploration, detailed in 'DatasetExploration.py'

    :param data: pd.DataFrame
    :return data: pd.DataFrame
    """
    mask = data['word_tokens'].apply(lambda x: True if len(x) > 15 else False)
    data = data[mask]
    return data


def main(sizes: list, type_: string):
    if not os.path.isdir('./Data/preprocessed'):
        os.makedirs('./Data/preprocessed')
    for size in sizes:
        data = load_dataset('./Data/unprocessed_' + size + '.h5')
        # preprocessing for fake news detection
        if type_ == 'fnd':
            print("Preparing Dataset for Fake News Clustering Experiments")
            data['unprocessed_text'] = data['text']

            print("Converting to lowercase")
            data = lower_case_text(data)

            print("Removing Punctuation")
            data = remove_punctuation(data, full_stop=False)

            print("Removing NewLines")
            data = remove_newlines(data)

            print("Removing Digits")
            data = remove_digits(data)

            print("Remvoing Stop words")
            data = remove_stop_words(data)

            # split data into two different providers
            print("Splitting into providers")
            providers = pd.unique(data['provider']).tolist()
            for provider in providers:
                data_provider = data[data['provider'] == provider]
                print("Removing articles with two labels")
                data_provider = remove_two_labeled(data_provider)
                print("Removing the noisey duplicates")
                data_provider = remove_noisey_duplicates(data_provider)
                print("Removing legitimate duplicates, keeping the first occurence")
                data_provider = remove_dupicated_articles(data_provider)
                print('Tokenizing Sentences')
                data_provider = tokenize_sentences(data_provider)
                print('Removing Final punctuation')
                data_provider = remove_punctuation(
                    data_provider, full_stop=True)
                print("Converting labels to categorical values")
                data_provider = convert_labels_to_categorical(data_provider)
                print("Tokenizing text")
                data_provider = tokenize_words(data_provider)
                data_provider = remove_articles_explored(data_provider)
                print("Done! Now saving data to './Data/Preprocessed'")
                joblib.dump(data_provider, './Data/Preprocessed/' +
                            provider + '_' + type_ + '_' + size + '.h5')

        # preprocessing for clustering
        elif type_ == 'clustering':
            print("Preparing Dataset for Topic Clustering Experiments")
            data['unprocessed_text'] = data['text']

            print("Converting to lowercase")
            data = lower_case_text(data)

            print("Removing Punctuation")
            data = remove_punctuation(data, full_stop=False)

            print("Removing NewLines")
            data = remove_newlines(data)

            print("Removing Digits")
            data = remove_digits(data)

            print("Remvoing Stop words")
            data = remove_stop_words(data)
            print("Removing Cookied articles")
            data = remove_cookied_articles(data)
            # split data into two different providers
            print("Splitting into providers")
            providers = pd.unique(data['provider']).tolist()
            for provider in providers:
                data_provider = data[data['provider'] == provider]

                print("Removing articles with two labels")
                data_provider = remove_two_labeled(data_provider)
                print("Removing the noisey duplicates")
                data_provider = remove_noisey_duplicates(data_provider)
                print("Removing legitimate duplicates, keeping the first occurence")
                data_provider = remove_dupicated_articles(data_provider)
                print('Tokenizing Sentences')
                data_provider = tokenize_sentences(data_provider)
                print('Removing Final punctuation')
                data_provider = remove_punctuation(
                    data_provider, full_stop=True)
                print("Stemming Words")
                data_provider = stem_words(data_provider)
                print("lemmatizing Words")
                data_provider = lemmatize_word(data_provider)
                print("Converting labels to categorical values")
                data_provider = convert_labels_to_categorical(data_provider)
                print("Tokenizing text")
                print("Cleaning URL")
                data_provider['website_name'] = data_provider['url'].apply(
                    lambda x: clean_url(x))
                print("Tokenizing text")
                data_provider = tokenize_words(data_provider)
                data_provider = remove_articles_explored(data_provider)
                print("Done! Now saving data to './Data/Preprocessed/" +
                      provider + '_' + type_ + '_' + size + '.h5')
                joblib.dump(data_provider, './Data/Preprocessed/' +
                            provider + '_' + type_ + '_' + size + '.h5')
