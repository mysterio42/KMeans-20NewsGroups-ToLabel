import glob
import os
import random
import string
from collections import defaultdict
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.data import to_json,to_csv
from utils.data import tokenizer

EMBEDDING_DIR = 'embedding/'
WEIGHTS_DIR = 'weights/'


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def dump_model(model):
    model_name = WEIGHTS_DIR + 'K-means-' + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def dump_embedding(embedding):
    path = EMBEDDING_DIR + 'tf-idf-' + generate_model_name(5) + '.pkl'

    with open(path, 'wb') as f:
        joblib.dump(value=embedding, filename=f, compress=3)
        print(f'Embedding saved at {path}')


def load_embedding(path):
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def latest_modified_embedding():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    embedding_files = glob.glob(EMBEDDING_DIR + '*')
    latest = max(embedding_files, key=os.path.getctime)
    return latest


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """

    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


class LabelMe:

    def __init__(self, sentences, n_clusters):
        """

        :param sentences: List of sentences
        :param n_clusters: cluster size
        """
        self.data = sentences
        self.embedding = TfidfVectorizer(tokenizer=tokenizer, lowercase=True)
        self.model = KMeans(n_clusters=n_clusters)

    def embed(self):
        tfidf_vectors = self.embedding.fit_transform(self.data)
        dump_embedding(self.embedding)
        return tfidf_vectors

    def train(self, embeds):
        self.model.fit(embeds)
        dump_model(self.model)

    def clusterize(self,fname):
        basename = fname + '-labeled'
        clusters = defaultdict(list)
        for i, label in enumerate(self.model.labels_):
            clusters['cluster_' + str(label)].append(self.data[i])
        labeled_data = dict(clusters)
        to_json(basename, labeled_data)
        to_csv(basename, labeled_data)
        del clusters
        return labeled_data
