import numpy as np
import pandas as pd
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


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


def vectorize(data, vectorizer):
    for i, X in enumerate(data):
        data[i] = vectorizer.transform(X).toarray()
    return data


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

    unigram_vectorizer = CountVectorizer(binary=True, max_features=max_features).fit(X_train)
    bigram_vectorizer = CountVectorizer(ngram_range=(2,2),binary=True, max_features=max_features).fit(X_train)

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
