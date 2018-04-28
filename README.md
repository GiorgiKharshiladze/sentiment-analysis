## Sentiment Analysis - Lab E in CS365 - Artificial Intelligence & Machine Learning Class :+1:

This is an implementation of the **Naive Bayes Classifier** written in python

- Our classifier uses words as features, adds the logprob scores for each token, and makes a binary decision between positive and negative. We also explored the effects of stop-word filtering. Removing common words like "the", "a" and "it" from our train and test sets.

- We use the stop-world list found at data/english.stop

- We also use n-fold cross-validation to test, which involves dividing the data into several sections (10 in this case), then training and testing the classifier repeatedly, with a different section as the held-out test set each time. Final accuracy is the average of the 10 runs which is at least 80%. 

- We use a movie review for training and the fact that it is positive or negative (the hand-labeled "true class") to help compute the correct statistics. When the same review is used for testing, we only use this label to compute your accuracy.


## Basic Call Structure and command line arguments

In the command line `cd` into the python directory and run: `python3 NaiveBayes.py -{FLAGNAME} ../data/imdb1`

For running FILTER_STOP_WORD, use `-f`, for BOOLEAN - True use `-b`, and for NEGATION - `-n`.
Examples:

1. `python3 NaiveBayes.py -f ../data/imdb1`
2. `python3 NaiveBayes.py -b ../data/imdb1`
3. `python3 NaiveBayes.py -n ../data/imdb1`


Authors: [Giorgi Kharshiladze](https://github.com/GiorgiKharshiladze/) & [Abhinav Pandey](https://github.com/abhinavp246)