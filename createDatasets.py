import json
import os
import pandas as pd
import joblib


def load_politifact_large_and_gossip_cop():
    path_to_data = "./Dataset"
    data = pd.DataFrame(
        columns=['article_index', 'provider', 'url', 'title', 'text', 'label'])
    # loop through providers of news
    for provider in os.listdir(path_to_data):
        if not provider.startswith('.'):
            # loop through real/fake sets of news
            for news_type in os.listdir(path_to_data + "/" + provider):
                if not news_type.startswith('.'):
                    # loop through each article folder
                    for articles in os.listdir(path_to_data + "/" + provider + "/" + news_type):
                        # open json file in folder
                        try:
                            with open(
                                    path_to_data + "/" + provider + '/' + news_type +
                                '/' + articles + "/news content.json",
                                    'r') as read_file:
                                file = json.load(read_file)
                                index = len(data)
                                data.loc[index] = [articles, provider, file['url'],
                                                   file['title'], file['text'], news_type]
                        except:
                            pass
    data = data[data['text'].map(len) > 0]
    if not os.path.isdir('./Data'):
        os.makedirs('./Data')
    joblib.dump(data, './Data/unprocessed_large.h5')


def load_politifact_small_and_buzzfeed():
    path_to_data = './Dataset2'
    data = pd.DataFrame(
        columns=['article_index', 'provider', 'url', 'title', 'text', 'label'])
    # loop through providers of news
    for provider in os.listdir(path_to_data):
        # avoid hidden folders
        if not provider.startswith('.'):
            # loop through real/fake sets of news
            for news_type in os.listdir(path_to_data + "/" + provider):
                if not news_type.startswith('.'):
                    # loop through each webpage
                    for webpage in os.listdir(path_to_data + "/" + provider + "/" + news_type):
                        # open json file
                        try:
                            with open(
                                    path_to_data + "/" + provider + '/' + news_type + '/' + webpage,
                                    'r') as read_file:
                                file = json.load(read_file)
                                index = len(data)
                                data.loc[index] = [webpage, provider, file['url'],
                                                   file['title'], file['text'], news_type]
                        except:
                            pass
    data = data[data['text'].map(len) > 0]
    joblib.dump(data, './Data/unprocessed_small.h5')


def main():
    load_politifact_large_and_gossip_cop()
    load_politifact_small_and_buzzfeed()


if __name__ == "__main__":
    main()
