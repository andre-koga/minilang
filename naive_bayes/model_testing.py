# used for checking the performance of the model on the test data
# and storing the results in a file.

from IO import load_training_data, load_model, MODEL_PATH
from language_code import get_language_name
import dill as pickle
from model_training import NaiveBayesClassifier

if __name__ == '__main__':
    main()
    
def PredictLanguage(word):
    naive_bayes_model = load_model()
    data = load_training_data()

    if naive_bayes_model is None:
        print("Model file is missing or empty. Training a new model.")
        
        naive_bayes_model = NaiveBayesClassifier()
        naive_bayes_model.train(data)
        
        with open(MODEL_PATH, 'wb') as file:
            pickle.dump(naive_bayes_model, file)

        print(f"Model saved to {MODEL_PATH}")

    predicted_language = naive_bayes_model.predict(word)

    print(f'The predicted language for the word "{word}" is: {get_language_name(predicted_language)}')
    
def main():
    word = input("Enter a word: ")
    PredictLanguage(word)