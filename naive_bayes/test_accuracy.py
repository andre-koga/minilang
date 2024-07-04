import os
import json
import random


def test_accuracy(test_data, model, accuracy):
    """
    Tests the accuracy of the model on the given test data.

    Parameters:
    - test_data (dict): A dictionary where keys are language labels and values are lists of strings in that language.
    - model (NaiveBayesClassifier): The trained NaiveBayesClassifier model.

    Returns:
    - dict: number of words and their accuracy when tested
    """
    total_count = 0
    correct_count = 0
    n_times = int(sys.argv[1]) #put in arguments for sys.args how many times to run,
    accuracy = {}

    for number in range(10): #test for 1 to 10 words
        for _ in range(n_times): #test n times for each word
            random_sentence_and_language = extract_random_word_and_language(number) # get (randomsentence, the actual language)
            random_sentence = [[' '.join(i)] for i in random_sentence_and_language[0]]
            predicted_language = model.predict(random_sentence) # use model to predict sentence and guess the language
            if predicted_language == random_sentence_and_language[1]: #compare them
                correct_count += 1
                total_count += 1
        acc_stats = correct_count / total_count if total_count > 0 else 0
        accuracy[number] = acc_stats #compute accuracy

    print(accuracy) #print and return the accuracy
    return accuracy



WORD_LIST_DIRECTORY = 'word_list'

def extract_random_word_and_language():

    (sentence, actual_language) = ([],"") # store randomword and actual language to return
    for _ in range(number_of_words):
    # List all JSON files in the directory
    return (sentence, actual_language)