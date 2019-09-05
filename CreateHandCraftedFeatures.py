'''Code implementing part of Semantics-level features proposed by 
    Zhou, X., Jain, A., Phoha, V. V., & Zafarani, R. (2019, April). Fake News Early Detection:
    A Theory-driven Model. arXiv:1904.11679 [cs]. Retrieved 2019-06-25, from http://arxiv.org/abs/1904.11679 
'''
import csv
import joblib
import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import normalize
from nltk.tokenize.simple import CharTokenizer
import numpy as np
import textstat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def create_csv_for_liwc(input_files):
    for file in input_files:
        temp = joblib.load('./results/'+file +
                           '/TopicClustering/lda_topic_data_5.h5')
        temp['unprocessed_text'] = temp['unprocessed_text'].apply(
            lambda x: " ".join(re.sub(r'[^\w\s!?.]', '', x).splitlines()))
        temp['unprocessed_text'] = temp['unprocessed_text'].apply(
            lambda x: re.sub(r'[-]', " ", x))
        temp[['label', 'cluster', 'unprocessed_text']].to_csv(
            './Data/LIWC/' + file + '.csv', sep=',', encoding='utf-8', header=False, index=False)


def get_data(input_files):
    output_files = []
    for file in input_files:
        print('Reading in:', file)
        output_files.append(pd.read_csv('./Data/LIWC/' + file + '_liwc.csv'))
    return output_files


def tokenize_text(data):
    try:
        data['word_tokens'] = data[['Source (A)', 'Source (C)']].apply(
            lambda x: word_tokenize(re.sub(r'[^\w\s]', '', x[1])), axis=1)
        data.rename(columns={'Source (A)': 'label',
                             'Source (B)': 'cluster',
                             'Source (C)': 'unprocessed_text'}, inplace=True)
    except KeyError:
        try:
            data['word_tokens'] = data[['A', 'C']].apply(
                lambda x: word_tokenize(re.sub(r'[^\w\s]', '', x[1])), axis=1)
            data.rename(
                columns={'A': 'label', 'B': 'cluster', 'C': 'unprocessed_text'}, inplace=True)
        except:
            data['word_tokens'] = data.iloc[:, 1].apply(
                lambda x: word_tokenize(re.sub(r'[^\w\s]', '', x)))
    return data


def get_informality_features(input_file, output):
    output['swear_words_num'] = (input_file['swear'] / 100) * input_file['WC']
    output['swear_words_prop'] = input_file['swear']
    output['netspeak_num'] = (input_file['netspeak'] / 100) * input_file['WC']
    output['netspeak_prop'] = input_file['netspeak']
    output['assent_num'] = (input_file['assent'] / 100) * input_file['WC']
    output['assent_prop'] = input_file['assent']
    output['nonfluenceies_num'] = (
        input_file['nonflu'] / 100) * input_file['WC']
    output['nonfluenceies_prop'] = input_file['nonflu']
    output['fillers_num'] = (input_file['filler'] / 100) * input_file['WC']
    output['fillers_prop'] = input_file['filler']
    output['overall_informal_num'] = (
        input_file['informal'] / 100) * input_file['WC']
    output['overall_informal_prop'] = input_file['informal']

    return output


def get_diversity_features(input_file, output):
    """Creates features using self-implementation and NLTK POS tagger
    Features are:
    # /% unique words

    Arguments:
        data {[type]} -- [description]
    """
    output['unique_word_prop'] = input_file['word_tokens'].apply(
        lambda x: len(pd.unique(x)) / len(x))
    output['unique_word_num'] = input_file['word_tokens'].apply(
        lambda x: len(pd.unique(x)))

    # Tag Text
    input_file['pos_tags'] = input_file['word_tokens'].apply(
        lambda x: pos_tag(x))

    # Unique Nouns
    output['unique_nouns_num'] = input_file['pos_tags'].apply(
        lambda x: len(pd.unique([i[0] for i in x if i[1] == 'NN' or i[1] == 'NNS'])))
    output['unique_nouns_prop'] = output[['WC', 'unique_nouns_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    # Unique Vowels
    output['unique_verbs_num'] = input_file['pos_tags'].apply(lambda x: len(
        pd.unique([i[0] for i in x if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])))
    output['unique_verbs_prop'] = output[['WC', 'unique_verbs_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    # Unique Adjectives
    output['unique_adjectives_num'] = input_file['pos_tags'].apply(lambda x: len(
        pd.unique([i[0] for i in x if i[1] in ['JJ', 'JJR', 'JJS']])))
    output['unique_adjectives_prop'] = output[[
        'WC', 'unique_adjectives_num']].apply(lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    # Unique Adverbs
    output['unique_adverbs_num'] = input_file['pos_tags'].apply(lambda x: len(
        pd.unique([i[0] for i in x if i[1] in ['RB', 'RBR', 'RBS']])))
    output['unique_advers_prop'] = output[['WC', 'unique_adverbs_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)
    return output


def get_subjectivity_features(input_file, output):
    with open('./bias-lexicon/bias-lexicon.txt') as f:
        biased_lexicons = f.read().splitlines()
    with open('./bias-lexicon/report_verbs.txt') as f:
        report_verbs = f.read().splitlines()
    with open('./bias-lexicon/factives_hooper1975.txt') as f:
        factive_verbs = f.read().splitlines()

    # Biased Lexicons
    output['biased_lex_num'] = input_file['word_tokens'].apply(
        lambda x: len([word for word in x if word in biased_lexicons]))
    output['biased_lex_prop'] = output[['WC', 'biased_lex_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    # Report Verbs
    output['report_verbs_num'] = input_file['word_tokens'].apply(
        lambda x: len([word for word in x if word in report_verbs]))
    output['report_verbs_prop'] = output[['WC', 'report_verbs_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    # Factive Verbs
    output['factive_verbs_num'] = input_file['word_tokens'].apply(
        lambda x: len([word for word in x if word in factive_verbs]))
    output['factive_verbs_prop'] = output[[
        'WC', 'factive_verbs_num']].apply(lambda x: round((x[1] / x[0]) * 100, 2), axis=1)

    return output


def get_sentiment_features(input_file, output):
    # Positive words
    output['positive_num'] = (input_file['posemo'] / 100) * input_file['WC']
    output['positive_prop'] = input_file['posemo']
    # Negative Words
    output['negative_num'] = (input_file['negemo'] / 100) * input_file['WC']
    output['negative_prop'] = input_file['negemo']
    # Anxiety words
    output['anxiety_num'] = (input_file['anx'] / 100) * input_file['WC']
    output['anxiety_prop'] = input_file['anx']
    # Anger Words
    output['anger_num'] = (input_file['anger'] / 100) * input_file['WC']
    output['anger_prop'] = input_file['anger']
    # Sadness Words
    output['sadness_num'] = (input_file['sad'] / 100) * input_file['WC']
    output['sadness_prop'] = input_file['sad']
    # Overall Emotional Words
    output['overall_emotional_num'] = (
        input_file['affect'] / 100) * input_file['WC']
    output['overall_emotional_prop'] = input_file['affect']
    senti_analyser = SentimentIntensityAnalyzer()
    # pass unprocessed text to sentiment analyser, but remove new lines and dashes (\n and -)
    output['average_sentiment_of_word'] = input_file['unprocessed_text'].apply(
        lambda x: senti_analyser.polarity_scores(" ".join(re.sub(r'[^\w\s!?.]', "", x).splitlines()))['compound'])
    return output


def get_quantity_features(input_file, output):
    char_tok = CharTokenizer()
    output['num_characters'] = input_file['unprocessed_text'].apply(
        lambda x: len(char_tok.tokenize(re.sub(' ', '', x))))
    output['num_words'] = input_file['WC']
    output['num_sentences'] = input_file['unprocessed_text'].apply(
        lambda x: len(sent_tokenize(x)))
    output['num_paragraphs'] = input_file['unprocessed_text'].apply(
        lambda x: len(x.split("\n")))
    output['avg_len_word'] = input_file['word_tokens'].apply(
        lambda x: round(np.mean([len(word) for word in x])), 2)
    output['avg_len_sentence'] = input_file['unprocessed_text'].apply(
        lambda x: np.mean([len([word for word in word_tokenize(re.sub(r'[.]', '', sentences))])
                           for sentences in sent_tokenize(x)]))
    output['avg_sent_per_para'] = input_file['unprocessed_text'].apply(
        lambda x: np.mean([len(sent_tokenize(sentences)) for sentences in x.split('\n\n')]))
    return output


def get_cognitive_process_features(input_file, output):
    output['insight_num'] = (input_file['insight'] / 100) * input_file['WC']
    output['insight_prop'] = input_file['insight']
    output['causation_num'] = (input_file['cause'] / 100) * input_file['WC']
    output['causation_prop'] = input_file['cause']
    output['discrepency_num'] = (
        input_file['discrep'] / 100) * input_file['WC']
    output['discrepency_prop'] = input_file['discrep']
    output['tentativeness_num'] = (
        input_file['tentat'] / 100) * input_file['WC']
    output['tentativeness_prop'] = input_file['tentat']
    output['certainty_num'] = (input_file['certain'] / 100) * input_file['WC']
    output['certainty_prop'] = input_file['certain']
    output['differentiation'] = (input_file['differ'] / 100) * input_file['WC']
    output['differentiation'] = input_file['differ']
    output['overall_cog_num'] = (
        input_file['cogproc'] / 100) * input_file['WC']
    output['overall_cog_prop'] = input_file['cogproc']
    return output


def get_perceptual_process_features(input_file, output):
    output['seeing_num'] = (input_file['see'] / 100) * input_file['WC']
    output['seeing_prop'] = input_file['see']
    output['hearing'] = (input_file['hear'] / 100) * input_file['WC']
    output['hearing'] = input_file['hear']
    output['feel'] = (input_file['feel'] / 100) * input_file['WC']
    output['feel'] = input_file['feel']
    output['overall_perc_num'] = (
        input_file['percept'] / 100) * input_file['WC']
    output['overall_perc_prop'] = input_file['percept']
    return output


def get_readability_features(input_file, output):
    output['FREI'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.flesch_reading_ease(x))
    output['FKGL'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.flesch_kincaid_grade(x))
    output['ARI'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.automated_readability_index(x))
    output['GFI'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.gunning_fog(x))
    output['CLI'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.coleman_liau_index(x))
    output['syllable_num'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.syllable_count(x))
    output['polysyllable_num'] = input_file['unprocessed_text'].apply(
        lambda x: textstat.textstat.polysyllabcount(x))
    output['long_words_num'] = input_file['unprocessed_text'].apply(
        lambda x: len([word for word in word_tokenize(x) if len(word) > 6]))
    return output


def get_punctuation_features(input_file, output):
    output['exclamation_mark_num'] = input_file['unprocessed_text'].apply(
        lambda x: len([exclam for exclam in x if exclam == '!']))
    output['question_mark_num'] = input_file['unprocessed_text'].apply(
        lambda x: len([question for question in x if question == '?']))
    output['ellipsis_num'] = input_file['unprocessed_text'].apply(
        lambda x: len([ellip for ellip in x if ellip == '...']))
    output['overall_punctuation'] = output[['exclamation_mark_num', 'question_mark_num', 'ellipsis_num']].apply(
        lambda x: x[0] + x[1] + x[2], axis=1)
    return output


def get_quality_features(input_file, output):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    output['stop_words_num'] = input_file['unprocessed_text'].apply(
        lambda x: len([stop for stop in word_tokenize(x) if stop in stop_words]))
    output['stop_words_proportion'] = output[['WC', 'stop_words_num']].apply(
        lambda x: round((x[1] / x[0]) * 100, 2), axis=1)
    return output


def classify(model, data):
    labels = data['label']
    data.drop(columns='label', inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=0)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Acc: ", accuracy_score(y_test, predictions))
    print("F1-Score: ", f1_score(y_test, predictions))
    print("Recall: ", recall_score(y_test, predictions))
    print("Precision: ", precision_score(y_test, predictions))


def main(force=False):
    import time
    input_files = ['politifact', 'gossipcop']

    if force == True:
        create_csv_for_liwc(input_files)
    print('Reading in Data Files')
    files = get_data(input_files)
    output_files = []
    for input_file in files:
        t0 = time.time()
        print('Creating word_tokens')
        input_file = tokenize_text(input_file)
        output = input_file[['label', 'cluster', 'WC']]
        print('Creating Informality Features')
        output = get_informality_features(input_file, output)
        print('Creating Diversity Features')
        output = get_diversity_features(input_file, output)
        print('Creating Subjectivity Features')
        output = get_subjectivity_features(input_file, output)
        print('Creating Sentiment Feautres')
        output = get_sentiment_features(input_file, output)
        print('Creating Quantity Features')
        output = get_quantity_features(input_file, output)
        print('Creating Cognitive Process Features')
        output = get_cognitive_process_features(input_file, output)
        print('Creating Perceptual Process Features')
        output = get_perceptual_process_features(input_file, output)
        print('Creating Readability Feautres')
        output = get_readability_features(input_file, output)
        print('Creating Punctuation Features')
        output = get_punctuation_features(input_file, output)
        print('Creating Quality Features')
        output = get_quality_features(input_file, output)
        print("Rounding features to 3 decimal places")
        output = np.round(output, decimals=3)
        output_files.append(output)
        print("Done features: ", time.time() - t0)
        print(output.head())
    print("Dumping files to './Data/HandCraftedFeatures/'")
    joblib.dump(output_files[0],
                './Data/HandCraftedFeatures/politifact_large.h5')
    joblib.dump(output_files[1], './Data/HandCraftedFeatures/gossipcop.h5')

    """joblib.dump(output_files[2], './Data/HandCraftedFeatures/BuzzFeed.h5')
    joblib.dump(output_files[3],
                './Data/HandCraftedFeatures/politifact_small.h5')"""





main()
