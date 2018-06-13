from functions import *
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.linear_model import Perceptron as Perc
from sklearn.svm import LinearSVC as SVC

"""
1  feature extraction:

compare unigrams, bigrams, word embeddings
get rid of first n most frequent features
find features that correlate the moest / semi-cross corelation
for one sentence binary or if exist (i.e. for NN ???)

2. categorization

Logistc Regression,
decision trees (forest)
K-NN
Neural Network


"""
if __name__ == "__main__":

    with open('Tweets-airline-sentiment.csv', 'rb') as f:
            dataset = pd.read_csv(f)

    if False:

        dataset_text = preprocess_set(dataset.text)
        dataset_lemma = preprocess_set(dataset.text, lemmatize=True)

        with open('pkl/dataset_text', 'wb') as f:
            pkl.dump(dataset_text, f)
        with open('pkl/dataset_lemma', 'wb') as f:
            pkl.dump(dataset_lemma, f)

    else:
        with open('pkl/dataset_text', 'rb') as f:
            dataset_text = pkl.load(f)
        with open('pkl/dataset_lemma', 'rb') as f:
            dataset_lemma = pkl.load(f)


    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(dataset_lemma, dataset.airline_sentiment)
    X_train, X_test, y_train, y_test = train_test_split(dataset_text, dataset.airline_sentiment)

    results = {}
    all_max_features = [100, 500, 1000, 2000, 4000, 8000]

    for max_features in all_max_features:

        vectorizers = get_ngram_vectorizers(X_train, max_features)
        X_train_ngrams = vectorize(vectorizers, X_train)
        X_test_ngrams = vectorize(vectorizers, X_test)

        vectorizers = get_ngram_vectorizers(X_train_l, max_features)
        X_train_ngrams_l = vectorize(vectorizers, X_train_l)
        X_test_ngrams_l = vectorize(vectorizers, X_test_l)


        models = [LogReg(random_state=43), NB(), # no random state for NB
            Perc(shuffle=True, random_state=45), SVC(random_state=46)]
        model_names = ['Logistic Regression', 'Naive Bayes', 'Perceptron', 'Linear SVM']


        for model, model_name in zip(models, model_names):
            print('Computing: {}'.format(model_name))
            for i in range(3):
                print('no lemmatization')
                model.fit(X_train_ngrams[i], y_train)
                score = model.score(X_test_ngrams[i], y_test)
                results[model_name + '_nl_' + str(max_features)] = score
                print('Accuracy for test set: ', score)

                print('with lematization')
                model.fit(X_train_ngrams_l[i], y_train)
                score = model.score(X_test_ngrams_l[i], y_test)
                results[model_name + '_wl_' + str(max_features)] = score
                print('Accuracy for test set: ', score)

    with open('pkl/grid-search_results.pkl', 'wb') as f:
        pkl.dump(results, f)
