from collections import Counter
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from numpy.random import seed
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.utils import class_weight
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from functions import *


def neural_network(X_train, X_test, y_train, y_test, model_name='', batch_size=32, epochs=300,
    learning_rate=0.01, patience=30):

    seed(42)
    set_random_seed(41)

    num_labels = 3

    # build model
    model = Sequential()

    model.add(Dense(50, input_shape=(X_train.shape[1],))) #256 ## , kernel_regularizer=regularizers.l2(0.1)
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(100))
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
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    early_callback = EarlyStopping(monitor='val_acc', patience=300)
    callbacks_list = [checkpoint, early_callback]

    
    # cw = class_weight.compute_class_weight('balanced', np.unique(binary_to_categorical(y_train)), binary_to_categorical(y_train))

    model_setup = {
        'batch_size': batch_size,
        'epochs': epochs,
        'validation_data': (X_test, y_test),
        'shuffle': True,
        # 'class_weight': cw,
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

    setup = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.002,
        'patience': 200,
    }

    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(dataset_lemma, dataset.airline_sentiment)
    X_train, X_test, y_train, y_test = train_test_split(dataset_text, dataset.airline_sentiment)

    y_train, y_test = encode_ys(y_train), encode_ys(y_test)

    neural_network(X_train, X_test, y_train, y_test, **setup)
