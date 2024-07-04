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

    print(f'The predicted language for the string {string} is: {get_language_name(predicted_language)}')

# -----------------------------------------------------------------

# currently only supports one argument
def main():
    if len(sys.argv) > 1:
        word = sys.argv[1:]
        PredictLanguage(word)
    # if len(sys.argv) > 2:
    #     word = sys.argv[1]
    #     size = sys.argv[2]
    #     PredictLanguage(word, size)
    # if len(sys.argv) > 3:
    #     word = sys.argv[1]
    #     size = int(sys.argv[2])
    #     path = sys.argv[3]
    #     PredictLanguage(word, size, path)
    # else:
    #     word = input("Enter a word: ")
    #     PredictLanguage(word)
    
if __name__ == '__main__':
    main()