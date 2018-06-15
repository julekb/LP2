from functions import *
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import Perceptron as Perc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC as SVC
import pandas as pd
import pickle as pkl


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

    # just vectors

    print('##########################')
    print('######## vectors #########')
    print('##########################')

    ys = LabelEncoder().fit_transform(dataset.airline_sentiment)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_text, ys, test_size=0.1)
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        dataset_lemma, ys, test_size=0.1)

    results = {}
    all_max_features = [100, 500, 1000, 2000, 4000]

    for max_features in all_max_features:

        print('Computing for {} max features'.format(max_features))

        vectorizers = get_ngram_vectorizers(X_train, max_features)
        X_train_ngrams = vectorize(vectorizers, X_train)
        X_test_ngrams = vectorize(vectorizers, X_test)

        vectorizers = get_ngram_vectorizers(X_train_l, max_features)
        X_train_ngrams_l = vectorize(vectorizers, X_train_l)
        X_test_ngrams_l = vectorize(vectorizers, X_test_l)


        models = [LogReg(random_state=43), RF(random_state=44),
            Perc(shuffle=True, random_state=45), SVC(random_state=46)]
        model_names = ['Logistic Regression', 'Random Forest', 'Perceptron', 'Linear SVM']

        for model, model_name in zip(models, model_names):
            print('Computing: {}'.format(model_name))
            for i in range(3):
                print('no lemmatization')
                model.fit(X_train_ngrams[i], y_train)
                score = model.score(X_test_ngrams[i], y_test)
                results[model_name + '_nl_' + str(max_features) + ['uni', 'bi', 'unibi'][i]] = score
                print('Accuracy for test set: ', score)

                print('with lematization')
                model.fit(X_train_ngrams_l[i], y_train)
                score = model.score(X_test_ngrams_l[i], y_test)
                results[model_name + '_wl_' + str(max_features) + ['uni', 'bi', 'unibi'][i]] = score
                print('Accuracy for test set: ', score)

    with open('pkl/grid-search_results_uni_bi.pkl', 'wb') as f:
        pkl.dump(results, f)


# word2vec

    print('##########################')
    print('######## word2vec ########')
    print('##########################')

    dataset_text_splited = split_dataset(dataset_text)
    dataset_lemma_splited = split_dataset(dataset_lemma)

    results = {}

    sizes = [100, 500, 1000, 2000, 4000]
    for size in sizes:

        print('Computing for {} size'.format(size))

        dataset_text = dataset_text_splited
        dataset_lemma = dataset_lemma_splited

        model_text = Word2Vec(dataset_text, size=size, seed=43)
        model_lemma = Word2Vec(dataset_lemma, size=size, seed=43)

        dataset_text = dataset2vec(dataset_text, model_text, size)
        dataset_lemma = dataset2vec(dataset_lemma, model_lemma, size)

        ys = LabelEncoder().fit_transform(dataset.airline_sentiment)
        X_train, X_test, y_train, y_test = train_test_split(
            dataset_text, ys, test_size=0.1)
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            dataset_lemma, ys, test_size=0.1)

        # models = [LogReg(random_state=43), RF(random_state=44),
        #     Perc(shuffle=True, random_state=45), SVC(random_state=46)]
        models = [LogReg(random_state=43),
            Perc(shuffle=True, random_state=45), SVC(random_state=46)]
        # model_names = ['Logistic Regression', 'Random Forest', 'Perceptron', 'Linear SVM']
        model_names = ['Logistic Regression', 'Perceptron', 'Linear SVM']

        for model, model_name in zip(models, model_names):
            print('Computing: {}'.format(model_name))
            print('no lemmatization')
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(size)
            results[model_name + '_nl_' + str(size)] = score
            print('Accuracy for test set: ', score)

            print('with lematization')
            model.fit(X_train_l, y_train)
            score = model.score(X_test_l, y_test)
            results[model_name + '_wl_' + str(size)] = score
            print('Accuracy for test set: ', score)

    with open('pkl/grid-search_results_word2vec.pkl', 'wb') as f:
        pkl.dump(results, f)
