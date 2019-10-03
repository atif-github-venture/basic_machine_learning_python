# Sourse -> https://www.datacamp.com/community/tutorials/simplifying-sentiment-analysis-python
# Another example to learn from -> https://medium.com/@ageitgey/text-classification-is-your-new-secret-weapon-7ca4fad15788
# https://www.kaggle.com/iwasdata/nltk-to-classify-adjectives-as-positive-negative



import nltk
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)


# Define the feature extractor

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))

# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(5)

print('That will be the end!')
