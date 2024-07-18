import os
import sys
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import IO
import language_code

class SVMLanguageDetector:
    def __init__(self, model_file_name='svm_language_detector.pkl'):
        self.model_file_name = model_file_name
        #we are using character n-grams ranging from 1-3
        #TfidfVectorizer gives every n-gram a TFIDF score
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        #giving labels integer values
        self.label_encoder = LabelEncoder()
        self.model = svm.LinearSVC()
        self.pipeline = make_pipeline(self.vectorizer, self.model)
        self.is_trained = False

    def load_data(self):
        data = IO.load_training_data()
        texts = []
        labels = []
        #making sure that every language has at least 2 samples, if not removing language
        for lang, words in data.items():
            if len(words) >= 2:
                for word in words:
                    texts.append(word)
                    labels.append(lang)
            else:
                print(f"Skipping language {lang} because of insufficient samples (only {len(words)} samples)")
        print(f"Loaded data for {len(set(labels))} languages: {set(labels)}")

        return texts, labels

    def train(self):
        texts, labels = self.load_data()
        y = self.label_encoder.fit_transform(labels)
        print(f"Labels: {self.label_encoder.classes_}")

        # Ensure that every class has at least 2 samples
        class_counts = np.bincount(y)
        valid_classes = np.where(class_counts >= 2)[0]
        valid_indices = [i for i, label in enumerate(y) if label in valid_classes]

        X_valid = [texts[i] for i in valid_indices]
        y_valid = y[valid_indices]

        print(f"Valid classes after filtering: {self.label_encoder.inverse_transform(valid_classes)}")
        print(f"Number of valid samples: {len(y_valid)}")

        #split the data for training and testing
        #stratify=y_valid to make sure training and testing have some proportion of class labels as the original dataset
        X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid)

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        IO.store_model(self.pipeline, file_base_name=self.model_file_name)
        print(f"Model saved as {self.model_file_name}")

    def load_model(self):
        full_path = os.path.join(IO.MODEL_DIRECTORY, self.model_file_name)
        self.pipeline = IO.load_model(full_path)
        if self.pipeline:
            self.is_trained = True
            print(f"Model loaded from {self.model_file_name}")
        else:
            print("Model loading failed")

    def predict_language(self, text):
        if not self.is_trained:
            raise Exception("Model is not trained. Please train or load the model first.")
        pred = self.pipeline.predict([text])
        return self.label_encoder.inverse_transform(pred)[0]

