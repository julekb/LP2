import pandas as pd
import numpy as np

from functions import strip_punctuation

"""
some general statistics
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

