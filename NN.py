from gensim.models import Word2Vec
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from numpy.random import seed
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
import numpy as np
import pickle as pkl

from functions import *


def neural_network(X_train, X_test, y_train, y_test, size, model_name='', batch_size=32, epochs=300, learning_rate=0.01, patience=30, layers_size=[200, 500]):

    seed(42)
    set_random_seed(41)

    num_labels = 3

    # build model
    model = Sequential()

    model.add(Dense(layers_size[0], input_shape=(size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(layers_size[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model_arguments = {
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'optimizer': optimizer,
    }

    model.compile(**model_arguments)

    # checkpoint

    filepath = 'models/NN-gridsearch-' + model_name + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
        save_best_only=True, mode='max')
    early_callback = EarlyStopping(monitor='val_acc', patience=300)
    callbacks_list = [checkpoint, early_callback]

    model_setup = {
        'batch_size': batch_size,
        'epochs': epochs,
        'validation_data': (X_test, y_test),
        'shuffle': True,
        'callbacks': callbacks_list,
        'verbose': 0
    }
    print('batch_size', model_setup['batch_size'])
    history = model.fit(X_train, y_train, **model_setup)

    return filepath, history


if __name__ == '__main__':

    # load dataset
    with open('Tweets-airline-sentiment.csv', 'rb') as f:
            dataset = pd.read_csv(f)
    with open('pkl/dataset_text', 'rb') as f:
        dataset_text = pkl.load(f)
    with open('pkl/dataset_lemma', 'rb') as f:
        dataset_lemma = pkl.load(f)

    ys = encode_ys(dataset.airline_sentiment)

    dataset_text_splited = split_dataset(dataset_text)
    dataset_lemma_splited = split_dataset(dataset_lemma)

    make_keras_picklable()

    grid, histories = [], []
    sizes = [100, 500, 1000]

    for size in sizes:

        dataset_text = dataset_text_splited
        dataset_lemma = dataset_lemma_splited

        model_text = Word2Vec(dataset_text, size=size, seed=43)
        model_lemma = Word2Vec(dataset_lemma, size=size, seed=43)

        dataset_text = dataset2vec(dataset_text, model_text, size)
        dataset_lemma = dataset2vec(dataset_lemma, model_lemma, size)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset_text, ys, test_size=0.1, random_state=47)
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
            dataset_lemma, ys, test_size=0.1, random_state=48)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        setup = {
            'batch_size': 256,
            'epochs': 150,
            'learning_rate': 0.002,
            'patience': 200,
            'layers_size': [size, 2 * size],
            'size': size
        }

        model_eval, history = neural_network(X_train, X_test, y_train, y_test, **setup)

        grid.append([model_eval])
        histories.append(history)

    save(histories, 'histories')
    save(grid, 'grid-search')
