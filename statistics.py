import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions import strip_punctuation, load

"""
some general statistics and plots
"""

if __name__ == "__main__":

    with open('Tweets-airline-sentiment.csv', 'rb') as f:
            dataset = pd.read_csv(f)

    all_numb = len(dataset)
    positive = sum(dataset.airline_sentiment == 'positive')
    neutral = sum(dataset.airline_sentiment == 'neutral')
    negative = sum(dataset.airline_sentiment == 'negative')

    print('Number of all datapoints:', all_numb)
    print('Percentage of positive: ', positive / all_numb * 100)
    print('Percentage of neutral: ', neutral / all_numb * 100)
    print('Percentage of negative: ', negative / all_numb * 100)

    # words

    print('')
    words = dataset.text.values
    words = strip_punctuation(words).split()
    print('Unique words in dataset: ', len(np.unique(words)))

    # plots for NN

    grid = load('grid-search')
    histories = load('histories')
    plt.clf()
    plt.figure(figsize=(5,15))
    batch_sizes = [8, 64, 128, 256]
    learning_rates = [0.1, .05, 0.01, 0.001]
    plt.figure(figsize=(15,10))
    plt.title('Neural Network accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    for history in histories:
        
        plt.plot(history.history['val_acc'], alpha=1)
    #     plt.plot(history.history['acc'], alpha=0.8)
    labels = ['size=100', 'size=500', 'size=1000']
    plt.legend(labels, loc='lower right')
    plt.savefig('nn-val-acc.jpg')

