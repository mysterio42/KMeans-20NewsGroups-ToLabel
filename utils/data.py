import json
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import glob, os

LABELED_DATA_DIR = 'labeled-data/'
DATA_DIR = 'data/'


def get_data(path):
    with open(path) as f:
        sentences = list(f.read().split('\n'))
    return sentences


def get_data_sources():
    return {os.path.splitext(os.path.basename(x))[0]: x for x in glob.glob(DATA_DIR + '*')}


def tokenizer(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens if token not in stopwords.words('english')]
    return tokens


def to_json(fname, data):
    with open(LABELED_DATA_DIR + fname + '.json', 'w') as f:
        json.dump(data, f)
    print(f'Data saved at {LABELED_DATA_DIR + fname}')


def to_csv(fname, data):
    df = pd.DataFrame.from_dict(data, orient='index')
    with open(LABELED_DATA_DIR + fname + '.txt', 'w') as f:
        f.write(df.to_markdown())
    df.to_csv(LABELED_DATA_DIR + fname + '.csv')
