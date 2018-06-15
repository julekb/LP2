from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from string import punctuation
import keras
import numpy as np
import pandas as pd
import pickle as pkl
import tempfile


def strip_punctuation(s, punctuation=punctuation):
    return ''.join(c for c in s if c not in punctuation + '”')


def preprocess_set(data, lemmatize=False):
    Xs = data.values.copy()

    for i in range(len(Xs)):

        Xs[i] = Xs[i].split(' ')
        Xs[i] = [x.lower() for x in Xs[i] if x != '']
        Xs[i] = [x for x in Xs[i] if x[0] != '@']
        if lemmatize:
            Xs[i] = [lemmatization(w) for w in Xs[i]]

        for j in range(len(Xs[i])):
            Xs[i][j] = strip_punctuation(Xs[i][j])

    return [' '.join(X) for X in Xs]


def NN(X_train, X_test, y_train, y_test):

    pass


def lemmatization(dataset):
    """
    not efficient way of lemmatization of a dataset
    """

    lemmatizer = WordNetLemmatizer()
    new_dataset = np.array([])
    for sent in dataset:
        new_sent = [lemmatizer.lemmatize(w) for w in sent]
        new_dataset = np.append(new_dataset, list([new_sent]))
    return new_dataset


if __name__ == "__main__":

    with open('Tweets-airline-sentiment.csv', 'rb') as f:
        dataset = pd.read_csv(f)

    dataset = dataset[:10]
    dataset_text = preprocess_set(dataset.text)
    dataset_lemma = preprocess_set(dataset.text, lemmatize=True)
    print(dataset_lemma.shape)
    print(dataset_text.shape)


def get_ngram_vectorizers(X_train, max_features=1000):

    unigram_vectorizer = CountVectorizer(binary=True,
        max_features=max_features).fit(X_train)
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),
        binary=True, max_features=max_features).fit(X_train)

    super_vectorizer = FeatureUnion([
        ('unigram', unigram_vectorizer),
        ('bigram', bigram_vectorizer)
    ])
    return unigram_vectorizer, bigram_vectorizer, super_vectorizer


def vectorize(vectorizers, dataset):

    vectorized = []
    for vectorizer in vectorizers:
        vectorized.append(vectorizer.transform(dataset))
    return vectorized


def encode_ys(ys_num):
    ys_num = LabelEncoder().fit_transform(ys_num)
    ys_num = LabelBinarizer().fit_transform(ys_num)
    return ys_num


def split_dataset(dataset):
    new_dataset = []
    for sent in dataset:
        new_dataset.append(sent.split(' '))
    return new_dataset


def sent2vec(sent, model, size=100):
    vec = np.zeros(size)
    word_num = len(sent)
    for word in sent:
        try:
            vec += model.wv[word]
        except KeyError:
            word_num -= 1
    if word_num != 0:
        return vec / word_num
    return vec


def dataset2vec(dataset, model, size=100):
    new_dataset = []
    for sent in dataset:
        new_dataset.append(sent2vec(sent, model, size))
    return new_dataset


def print_for_latex(results, sizes, models):
    """
    function for printing results in a appropiate form for latex table
    """
    for model in models:
        for t in ['nl', 'wl']:
            line = model + ' ' + t.upper()
            for size in sizes:
                key = model + '_' + t + '_' + str(size)
                line += ' & ' + str(round(results[key], 4))
            line += ' \\\hline'
            print(line)


def make_keras_picklable():
    """ source: http://zachmoshe.com/2017/04/03/pickling-keras-models.html"""

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def save(object, path):
    with open('pkl/' + path + '.pkl', 'wb') as f:
        pkl.dump(object, f)
    return


def load(path):

    with open('pkl/' + path + '.pkl', 'rb') as f:
        return pkl.load(f)
