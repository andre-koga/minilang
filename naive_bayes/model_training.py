# used for training the bayes model.

from IO import load_training_data, load_model, MODEL_PATH
from language_code import get_language_name
from collections import defaultdict
import numpy as np
import dill as pickle

# -----------------------------------------------------------------

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(self.default_feature_probs))
    
    @staticmethod
    def default_feature_probs():
        return defaultdict(float)
    
    def extract_features(word):
        features = defaultdict(int)
        # Count frequency of each letter
        for char in word:
            features[f'char_{char}'] += 1
        # Count frequency of each pair of neighboring letters (bigrams)
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            features[f'bigram_{bigram}'] += 1
        # Count frequency of each triplet of neighboring letters (trigrams)
        for i in range(len(word) - 2):
            trigram = word[i:i+3]
            features[f'trigram_{trigram}'] += 1
        # Add length of the word as a feature
        features['length'] = len(word)
        return features
        
    def train(self, data):
        total_words = sum(len(words) for words in data.values())
        vocab_size = len(set(word for words in data.values() for word in words))  # Estimate of total vocabulary size
        for language, words in data.items():
            self.class_probs[language] = len(words) / total_words
            for word in words:
                features = self.extract_features(word)
                for feature, value in features.items():
                    self.feature_probs[language][feature][value] += 1
        for language in self.feature_probs:
            for feature in self.feature_probs[language]:
                total = sum(self.feature_probs[language][feature].values())
                for value in self.feature_probs[language][feature]:
                    # Apply Laplace smoothing
                    self.feature_probs[language][feature][value] = (self.feature_probs[language][feature][value] + 1) / (total + vocab_size)
        
    def predict(self, word):
        features = self.extract_features(word)
        scores = {}
        for language in self.class_probs:
            scores[language] = np.log(self.class_probs[language])
            for feature, value in features.items():
                if value in self.feature_probs[language][feature]:
                    scores[language] += np.log(self.feature_probs[language][feature][value])
                else:
                    scores[language] += np.log(1e-6)  # Smoothing for unseen features
        return max(scores, key=scores.get)