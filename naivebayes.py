import pandas as pd
from collections import defaultdict
import numpy as np
from wordfreq import top_n_list, get_frequency_list, available_languages
import os
import json


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
        self.class_probs = defaultdict(float)
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
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

# Create and train the classifier
classifier = NaiveBayesClassifier()
classifier.train(data)

# Predict the language of a word
word = 'bonjour'
predicted_language = classifier.predict(word)

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

# Language    Code    #  Large?   WP    Subs  News  Books Web   Twit. Redd. Misc.
# ──────────────────────────────┼────────────────────────────────────────────────
# Arabic      ar      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Bangla      bn      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Bosnian     bs [1]  3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Bulgarian   bg      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Catalan     ca      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Chinese     zh [3]  7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   -     Jieba
# Croatian    hr [1]  3         │ Yes   Yes   -     -     -     Yes   -     -
# Czech       cs      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Danish      da      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Dutch       nl      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# English     en      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Finnish     fi      6  Yes    │ Yes   Yes   Yes   -     Yes   Yes   Yes   -
# French      fr      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# German      de      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Greek       el      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Hebrew      he      5  Yes    │ Yes   Yes   -     Yes   Yes   Yes   -     -
# Hindi       hi      4  Yes    │ Yes   -     -     -     Yes   Yes   Yes   -
# Hungarian   hu      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Icelandic   is      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Indonesian  id      3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Italian     it      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Japanese    ja      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Korean      ko      4  -      │ Yes   Yes   -     -     -     Yes   Yes   -
# Latvian     lv      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Lithuanian  lt      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Macedonian  mk      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Malay       ms      3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Norwegian   nb [2]  5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Persian     fa      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Polish      pl      6  Yes    │ Yes   Yes   Yes   -     Yes   Yes   Yes   -
# Portuguese  pt      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Romanian    ro      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Russian     ru      5  Yes    │ Yes   Yes   Yes   Yes   -     Yes   -     -
# Slovak      sk      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Slovenian   sl      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Serbian     sr [1]  3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Spanish     es      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Swedish     sv      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Tagalog     fil     3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Tamil       ta      3  -      │ Yes   -     -     -     Yes   Yes   -     -
# Turkish     tr      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Ukrainian   uk      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Urdu        ur      3  -      │ Yes   -     -     -     Yes   Yes   -     -
# Vietnamese  vi      3  -      │ Yes   Yes   -     -     Yes   -     -     -