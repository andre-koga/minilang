import os
import sys
import joblib
from IO import load_model, load_training_data, MAX_WORD_LIST_SIZE, MODEL_DIRECTORY
from sklearn.preprocessing import LabelEncoder
import Lang

class SVMLanguagePredictor:
    def __init__(self, model_file_name='svm_language_detector.pkl'):
        self.model_file_name = model_file_name
        self.model = None
        self.label_encoder = LabelEncoder()
        self.load_model()

    def load_model(self):
        full_path = os.path.join(MODEL_DIRECTORY, self.model_file_name)
        print(f"Attempting to load model from: {full_path}")

        if not os.path.exists(full_path):
            raise Exception(f"Model file does not exist at path: {full_path}")

        try:
            self.model = joblib.load(full_path)
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

        if self.model:
            print(f"Model loaded successfully from {full_path}")
            # Initialize the label encoder with the correct classes
            try:
                data = load_training_data(size=MAX_WORD_LIST_SIZE, weighted=False)
                all_labels = list(data.keys())
                self.label_encoder.fit(all_labels)
                print(f"Label encoder classes: {self.label_encoder.classes_}")
            except Exception as e:
                raise Exception(f"Error loading training data or initializing label encoder: {e}")
        else:
            raise Exception("Model loading failed")

    def predict_language(self, text):
        if not self.model:
            raise Exception("Model is not loaded. Please load the model first.")
        pred = self.model.predict([text])
        return self.label_encoder.inverse_transform(pred)[0]

def main():
    predictor = SVMLanguagePredictor()

    #test model by trying different sentences here 
    testing_sentence = 'आप के साथ क्या गलत हुआ है'
    predicted_lang = predictor.predict_language(testing_sentence)
    print(f"The predicted language is: {Lang.get_language_name(predicted_lang)}")

if __name__ == '__main__':
    main()
