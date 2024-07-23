from IO import load_training_data
from Lang import LANGUAGE_CODE

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import warnings

class RandomForestLanguageClassifier:
    def __init__(self, words_per_language = 1000, n_trees = 100, max_depth = 2, max_ngrams = 2, n_pca = -1):
        self.words_per_language = words_per_language
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_ngrams = max_ngrams
        self.n_pca = n_pca

    def train(self):
        """
        Train the random forest classifier on the data

        returns: None
        """
        data = load_training_data(self.words_per_language, only_non_logographic=True, weighted=True)
        print("Words loaded...")

        tuples = [(word, lang, freq) for lang in data for word, freq in data[lang].items()]
        df = pd.DataFrame(columns = ['Word', 'Language', 'Frequency'], data = tuples)
        
        self.encoder = LabelEncoder().fit(df['Language'])

        y = self.encoder.transform(df['Language'])
        X = self._generate_features(df['Word'])
        freqs = df['Frequency']
        self.cols = X.columns

        print("Training data generated...")

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if self.n_pca != -1:
                self.pca = PCA(n_components=self.n_pca)
                X = self.pca.fit_transform(X)
                print("Dimensionality reduced...")

            self.clf = RandomForestClassifier(verbose=2, n_estimators=self.n_trees,
                max_depth=self.max_depth, n_jobs=-1)
            self.clf.fit(X, y, sample_weight=freqs)
            self.clf.verbose = 0

    def _generate_features_for_word(self, word):
        """
        Get all features (ngrams) for a given word

        word: string to get features off
        returns: dictionary with the counts of each ngram
        """
        ngrams = {}
        for n in range(0, self.max_ngrams):
            for idx in range(0, len(word) - n):
                curr_ngram = word[idx:idx + n + 1]
                ngrams[curr_ngram] = ngrams.get(curr_ngram, 0) + 1
        return ngrams

    def _generate_features(self, words):
        """
        Convert a list of words to the features (ngrams) in the word

        word: arraylike object containing strings to extract features of
        returns: data: pd.DataFrame containing the count of each ngram
        """
        cols = set()
        ngrams_counts = []
        for word in words:
            ngrams = self._generate_features_for_word(word.lower())
            ngrams_counts.append(ngrams)
            cols.update(ngrams.keys())

        data = pd.DataFrame(columns = list(cols), data = ngrams_counts)
        data = data.fillna(0)

        return data
    
    def predict(self, sentence):
        """
        Uses the trained random forest classifier to predict the language of a sentence

        sentence: string containing words separated by spaces
        returns: tuple(string: hard prediction of the language of the sentence,
                       list[float]: soft prediction of the probability of each langauge)
        """
        words = sentence.split(" ")
        features = self._generate_features(words)
        X = pd.DataFrame(columns = self.cols)

        try:
            merged = X.merge(features, how='right')[self.cols].fillna(0)
        except:
            return "Unknown", [0] * len(LANGUAGE_CODE)


        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if self.n_pca != -1:
                merged = self.pca.transform(merged)
            preds = self.clf.predict_proba(merged)

        soft_prediction = np.array(preds).sum(axis=0)

        # below removes since not doing log probs
        # soft_prediction = np.exp(soft_prediction - np.max(soft_prediction))
        soft_prediction = np.exp(soft_prediction) / np.sum(np.exp(soft_prediction)) # softmax

        hard_prediction = self.encoder.classes_[np.argmax(soft_prediction)]

        return hard_prediction, soft_prediction

if __name__ == "__main__":
    from IO import store_model
    from sklearn import tree

    n_trees = 1000
    max_depth = 3
    n_pca = -1
    max_ngrams = 2

    model_name = f"{n_trees}t_{max_depth}d_{n_pca}p_{max_ngrams}n"

    model = RandomForestLanguageClassifier(1000, n_trees, max_depth, max_ngrams, n_pca)
    model.train()
    try:
        while True:
            word = input("Enter a sentence or 'quit': ")
            if word == 'quit':
                break
            prediction, probs = model.predict(word)
            print(f"Predicted language: {prediction}")
    finally:
        store_model(model, model_name)