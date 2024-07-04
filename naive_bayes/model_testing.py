# used for checking the performance of the model on the test data
# and storing the results in a file.

from IO import load_training_data, load_model, store_model, MODEL_BASE_PATH
from language_code import get_language_name
from model_training import NaiveBayesClassifier
import sys
    
def PredictLanguage(string, words_size=100000, ngrams=(1, 2, 3), path = MODEL_BASE_PATH, alter_base_path=True):
    # set alter_base_path to False to use the base path as is.
    # the directory is hardcoded on the IO.py file.
    full_path = f'{words_size}_{ngrams}_{path}' if alter_base_path else path
    naive_bayes_model = load_model(file_name=path)

    if naive_bayes_model is None:
        data = load_training_data(size=words_size)
        
        print(f'Model file is missing or empty. Training a new model using the top {words_size} words per language.')
        
        naive_bayes_model = NaiveBayesClassifier()
        naive_bayes_model.train(data, ngrams=ngrams)
        
        store_model(naive_bayes_model, file_base_name=full_path)

    predicted_language = naive_bayes_model.predict(string)

    print(f'The predicted language for the string "{string}" is: {get_language_name(predicted_language)}')
    print(f'Bear in mind that lowercase and uppercase may affect the prediction.')

# -----------------------------------------------------------------

def main():
    if len(sys.argv) == 2:
        # order of arguments:
        # 1. (string) string to predict the language of

        # an example of how to run the script:
        # python model_testing.py "this is a test string"
        
        string = sys.argv[1]
        
        PredictLanguage(string)
    elif len(sys.argv) == 4:
        # order of arguments:
        # 1. (string) string to predict the language of
        # 2. (optional int) number of words to use for training
        # 3. (optional tuple of ints) n-grams to use for training

        # an example of how to run the script:
        # python model_testing.py "this is a test string" 100000 "1,2,3"
        
        string = sys.argv[1]
        words_size = int(sys.argv[2])
        n_grams = tuple(map(int, sys.argv[3].split(',')))
        
        PredictLanguage(string, words_size, n_grams)
    elif len(sys.argv) == 6:
        # order of arguments:
        # 1. (string) string to predict the language of
        # 2. (optional int) number of words to use for training
        # 3. (optional tuple of ints) n-grams to use for training
        # 4. (optional string) path to the model file
        # 5. (optional bool) alter the base path to include the number of words and n-grams

        # an example of how to run the script:
        # python model_testing.py "this is a test string" 100000 "1,2,3" "model.pkl" False
        
        string = sys.argv[1]
        words_size = int(sys.argv[2])
        n_grams = tuple(map(int, sys.argv[3].split(',')))
        path = sys.argv[4]
        alter_base_path = bool(sys.argv[5])
        
        PredictLanguage(string, words_size, n_grams, path, alter_base_path)
    else:
        print('Invalid number of arguments. Please follow the instructions.')
        sys.exit(1)
    
if __name__ == '__main__':
    main()