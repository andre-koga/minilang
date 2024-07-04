# used for training the bayes model.

from collections import defaultdict
import numpy as np
import sys

# -----------------------------------------------------------------

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(self.default_feature_probs))
    
    @staticmethod
    def default_feature_probs():
        return defaultdict(float)
    
    def extract_features(self, string, ngrams=(1, 2, 3)):
        features = defaultdict(int)
        # Count frequency of each letter
        words = string.split()
        for word in words:
            for char in word:
                features[f'char_{char}'] += 1
            # Count the frequency of the chosen n-grams
            for n in ngrams:
                for i in range(len(word) - n + 1):
                    ngram = word[i:i+n]
                    features[f'ngram_{ngram}'] += 1
        # Add length of the word as a feature
            features['length'] += len(word)
        return features
        
    def train(self, data, ngrams=(1, 2, 3), weighted=False):
        if weighted:
            self.train_weighted(data, ngrams)
            return
        total_words = sum(len(words) for words in data.values())
        vocab_size = len(set(word for words in data.values() for word in words))  # Estimate of total vocabulary size
        for language, words in data.items():
            self.class_probs[language] = len(words) / total_words
            for word in words:
                features = self.extract_features(word, ngrams=ngrams)
                for feature, value in features.items():
                    self.feature_probs[language][feature][value] += 1
        for language in self.feature_probs:
            for feature in self.feature_probs[language]:
                total = sum(self.feature_probs[language][feature].values())
                for value in self.feature_probs[language][feature]:
                    # Apply Laplace smoothing
                    self.feature_probs[language][feature][value] = (self.feature_probs[language][feature][value] + 1) / (total + vocab_size)
    
    def train_weighted(self, data, ngrams=(1, 2, 3)):
        total_words = sum(freq for words_freq in data.values() for freq in words_freq.values())
        # total_words = sum(len(words) * freq for words, freq in data.values())
        vocab_size = len(set(word for words in data.keys() for word in words))  # Estimate of total vocabulary size
        for language, words_freq in data.items():
            self.class_probs[language] = sum(freq for freq in words_freq.values()) / total_words
            for word, freq in words_freq.items():
                features = self.extract_features(word, ngrams=ngrams)
                for feature, value in features.items():
                    # Multiply the feature count by the word frequency
                    self.feature_probs[language][feature][value] += value * freq
        for language in self.feature_probs:
            for feature in self.feature_probs[language]:
                total = sum(self.feature_probs[language][feature].values())
                for value in self.feature_probs[language][feature]:
                    # Apply Laplace smoothing
                    self.feature_probs[language][feature][value] = (self.feature_probs[language][feature][value] + 1) / (total + vocab_size)
                
    def predict(self, string):
        features = self.extract_features(string)
        scores = {}
        for language in self.class_probs:
            scores[language] = np.log(self.class_probs[language])
            for feature, value in features.items():
                if value in self.feature_probs[language][feature]:
                    scores[language] += np.log(self.feature_probs[language][feature][value])
                else:
                    scores[language] += np.log(1e-6)  # Smoothing for unseen features
        return max(scores, key=scores.get)