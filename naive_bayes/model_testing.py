# used for checking the performance of the model on the test data
# and storing the results in a file.

from IO import load_training_data, load_model, store_model, MODEL_PATH
from language_code import get_language_name
from model_training import NaiveBayesClassifier
import sys
    
def PredictLanguage(string, words_size=100000, path = MODEL_PATH):
    naive_bayes_model = load_model(file_name=path)
    data = load_training_data(size=words_size)

    if naive_bayes_model is None:
        print("Model file is missing or empty. Training a new model.")
        
        naive_bayes_model = NaiveBayesClassifier()
        naive_bayes_model.train(data)
        
        store_model(naive_bayes_model, file_name=path)

    predicted_language = naive_bayes_model.predict(string)

    print(f'The predicted language for the string "{string}" is: {get_language_name(predicted_language)}')
    print(f'Bear in mind that lowercase and uppercase may affect the prediction.')

# -----------------------------------------------------------------

# currently only supports a single argument, the string
def main():
    if len(sys.argv) == 2:
        string = sys.argv[1]
        PredictLanguage(string)
    else:
        print("Please provide a single argument, the string to predict the language of.")
        sys.exit(1)
    
if __name__ == '__main__':
    main()