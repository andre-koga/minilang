import pandas as pd
from collections import defaultdict
import numpy as np
from wordfreq import top_n_list, get_frequency_list, available_languages
import os
import json
import dill as pickle


# Directory to store word lists
word_list_dir = 'word_lists'

# Ensure the directory exists
os.makedirs(word_list_dir, exist_ok=True)

def save_word_list(lang, words):
    """Save the word list to a local file."""
    file_path = os.path.join(word_list_dir, f'{lang}.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(words, f)

def load_word_list(lang):
    """Load the word list from a local file, if it exists."""
    file_path = os.path.join(word_list_dir, f'{lang}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


data = {}
langs = available_languages(wordlist='best')
for lang in langs.keys():
    # Try to load from local file first
    words = load_word_list(lang)
    if words is None:
        # If not available locally, download and save
        words = top_n_list(lang, 100000, wordlist='best')
        save_word_list(lang, words)
    data[lang] = words


# Feature extraction function
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

# Train a Naive Bayes classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(self.default_feature_probs))
    
    @staticmethod
    def default_feature_probs():
        return defaultdict(float)
    
    # def train(self, data):
    #     total_words = sum(len(words) for words in data.values())
    #     for language, words in data.items():
    #         self.class_probs[language] = len(words) / total_words
    #         for word in words:
    #             features = extract_features(word)
    #             for feature, value in features.items():
    #                 self.feature_probs[language][feature][value] += 1
    #     for language in self.feature_probs:
    #         for feature in self.feature_probs[language]:
    #             total = sum(self.feature_probs[language][feature].values())
    #             for value in self.feature_probs[language][feature]:
    #                 self.feature_probs[language][feature][value] /= total
        
    def train(self, data):
        total_words = sum(len(words) for words in data.values())
        vocab_size = len(set(word for words in data.values() for word in words))  # Estimate of total vocabulary size
        for language, words in data.items():
            self.class_probs[language] = len(words) / total_words
            for word in words:
                features = extract_features(word)
                for feature, value in features.items():
                    self.feature_probs[language][feature][value] += 1
        for language in self.feature_probs:
            for feature in self.feature_probs[language]:
                total = sum(self.feature_probs[language][feature].values())
                for value in self.feature_probs[language][feature]:
                    # Apply Laplace smoothing
                    self.feature_probs[language][feature][value] = (self.feature_probs[language][feature][value] + 1) / (total + vocab_size)
        
    def predict(self, word):
        features = extract_features(word)
        scores = {}
        for language in self.class_probs:
            scores[language] = np.log(self.class_probs[language])
            for feature, value in features.items():
                if value in self.feature_probs[language][feature]:
                    scores[language] += np.log(self.feature_probs[language][feature][value])
                else:
                    scores[language] += np.log(1e-6)  # Smoothing for unseen features
        return max(scores, key=scores.get)

def load_model(file_name):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        # File does not exist or is empty
        return None
    try:
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
            return model
    except (EOFError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        return None

# Create and train the classifier
naive_bayes_model = load_model('naive_bayes_model.pkl')
if naive_bayes_model is None:
    print("Model file is missing or empty. Training a new model.")
    naive_bayes_model = NaiveBayesClassifier()
    naive_bayes_model.train(data)
    model_file_path = 'naive_bayes_model.pkl'
    with open(model_file_path, 'wb') as file:
        pickle.dump(naive_bayes_model, file)

    print(f"Model saved to {model_file_path}")


# Predict the language of a word
word = 'bonjour'
predicted_language = naive_bayes_model.predict(word)

language_code = {
    'ar': 'Arabic',
    'bn': 'Bangla',
    'bs': 'Bosnian',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ms': 'Malay',
    'nb': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sr': 'Serbian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'fil': 'Tagalog',
    'ta': 'Tamil',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese'
}

print(f'The predicted language for the word "{word}" is: {language_code[predicted_language]}')