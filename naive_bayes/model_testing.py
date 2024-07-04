# used for checking the performance of the model on the test data
# and storing the results in a file.

from IO import load_training_data, load_model, store_model, MODEL_BASE_PATH, MAX_WORD_LIST_SIZE
from language_code import get_language_name
from model_training import NaiveBayesClassifier
import sys
    
def PredictLanguage(string, words_size=MAX_WORD_LIST_SIZE, ngrams=(1, 2, 3), weighted=False, path = MODEL_BASE_PATH, alter_base_path=True):
    # set alter_base_path to False to use the base path as is.
    # the directory is hardcoded on the IO.py file.
    full_path = f'{words_size}_{ngrams}_{weighted}_{path}' if alter_base_path else path
    naive_bayes_model = load_model(file_name=full_path)

    if naive_bayes_model is None:
        data = load_training_data(size=words_size, weighted=weighted)
        
        print(f'Model file is missing or empty. Training a new model using the arguments: words_size={words_size}, ngrams={ngrams}, weighted={weighted}.')
        print(f'It will be stored at {full_path}.')
        
        naive_bayes_model = NaiveBayesClassifier()
        naive_bayes_model.train(data, ngrams=ngrams, weighted=weighted)
        
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
        
        # IMPORTANT! this by default uses the top MAX_WORD_LIST_SIZE words and n-grams (1, 2, 3) and weighted=False
        
        string = sys.argv[1]
        
        PredictLanguage(string)
    elif len(sys.argv) == 5:
        # order of arguments:
        # 1. (string) string to predict the language of
        # 2. (optional int) number of words to use for training
        # 3. (optional tuple of ints) n-grams to use for training
        # 4. (optional bool) weight words based on frequency?

        # an example of how to run the script:
        # python model_testing.py "this is a test string" 100 "1,2,3" False
        
        string = sys.argv[1]
        words_size = int(sys.argv[2])
        n_grams = tuple(map(int, sys.argv[3].split(',')))
        weighted = False if sys.argv[4] == 'False' else True
        
        PredictLanguage(string, words_size, n_grams, weighted)
    elif len(sys.argv) == 7:
        # order of arguments:
        # 1. (string) string to predict the language of
        # 2. (optional int) number of words to use for training
        # 3. (optional tuple of ints) n-grams to use for training
        # 4. (optional bool) weight words based on frequency?

        # 5. (optional string) path to the model file
        # 6. (optional bool) alter the base path to include the number of words and n-grams

        # an example of how to run the script:
        # python model_testing.py "this is a test string" 100 "1,2,3" False "model.pkl" False
        
        string = sys.argv[1]
        words_size = int(sys.argv[2])
        n_grams = tuple(map(int, sys.argv[3].split(',')))
        weighted = bool(sys.argv[4])
        
        path = sys.argv[5]
        alter_base_path = bool(sys.argv[6])
        
        PredictLanguage(string, words_size, n_grams, weighted, path, alter_base_path)
    else:
        print('Invalid number of arguments. Please follow the instructions.')
        sys.exit(1)
    
if __name__ == '__main__':
    main()