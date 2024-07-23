# Tests the accuracy of the model on a given dataset.

import os
import json
import random
from IO import *
from model_training import RandomForestLanguageClassifier

#-------------------------------------------------------------

STAT_DIR = 'stats'

#-------------------------------------------------------------

def test_accuracy(n_times : int, word_lengths : list[str], model, name) -> dict:
    """
    Evaluate a given model on n_times number of words for each language, for sentences
    of each length in word_lengths. Save the results to a file in the stats directory.

    Parameters:
    - n_times: number of words per language to test
    - word_lengths: list of word lengths to test
    - model_path: path to the model file

    Returns: dictionary of the form {language: {word_length: accuracy, ...}, ...}
    """
    accuracy = {}

    for number in word_lengths: # how long should the sentence be
        training_data = generate_test_dataset(number, n_times)

        for language in training_data:
            total_count = 0
            correct_count = 0

            print(language, end = " ")

            for sentence in training_data[language]:
                total_count += 1
                prediction, _ = model.predict(" ".join(sentence))
                if prediction == language:
                    correct_count += 1

            accuracy[language] = accuracy.get(language, {})
            accuracy[language][number] = correct_count / total_count

            print(accuracy[language])

        print("Finished testing for word length: ", number)

    print(accuracy) #print and return the accuracy

    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)

    with open(os.path.join(STAT_DIR, name + '.json'), 'w') as f:
        json.dump(accuracy, f, indent=4)

    return accuracy
#-------------------------------------------------------------

def generate_test_dataset(words_per_language, number_of_words):
    """
    Creates a dataset of random words for each language for model testing.

    Parameters:
    - words_per_language: number of words per language
    - number_of_words: number of languages to test

    Returns: dictionary of the form {language: [[word1, word2, ...], ...], ...}.
    Note that there will be words_per_language * number_of_words * number_of_languages (~40) words in total.
    """
    languages_dict = load_training_data(size = MAX_WORD_LIST_SIZE, weighted=False)
    languages_dict = {lang : words for lang, words in languages_dict.items() if lang in non_logographic_langs}
    
    out = {}
    for language in non_logographic_langs:
        out[language] = [random.choices(languages_dict[language], k=words_per_language)
                            for _ in range(number_of_words)]

    return out

#-------------------------------------------------------------
def main():
    import argparse

    # example usage: python ./test_accuracy.py MODEL_NAME -n 50 -w 1 2 5

    parser = argparse.ArgumentParser(description='Test the accuracy of the model')
    parser.add_argument("model_path", type=str, help='Path to the model file')
    parser.add_argument('-n', "--num_tests", type=int, help='Number of words per language for testing (default 50)', required=False, default=50)
    parser.add_argument('-w', "--word_lengths", type=str, help='Word lengths to test the model on (default = (1, 2,5))', required=False, default=["1", "2", "5"], nargs='+')

    args = parser.parse_args()

    n_times = int(args.num_tests)
    word_lengths = [int(i.replace('(', '').replace(')', '').replace(',', '')) for i in args.word_lengths]
    model_path = args.model_path

    assert min(word_lengths) > 0, "Word length must be greater than 0"

    model = load_model(model_path)
    test_accuracy(n_times, word_lengths, model, model_path)

if __name__ == '__main__':
    main()