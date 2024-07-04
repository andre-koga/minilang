import os
import json
import random
import sys
from IO import *
import subprocess

#-------------------------------------------------------------

def test_accuracy(n_times):
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
    accuracy = {}

    for number in range(1, 11): #test for 1 to 10 words
        for _ in range(n_times): #test n times for each word
            random_sentence_and_language = extract_random_word_and_language(number) # get (randomsentence, the actual language)
            if number < 1:
                random_sentence = [[' '.join(i)] for i in random_sentence_and_language[0]]
            elif number == 1:
                random_sentence = random_sentence_and_language[0]

            predicted_language = execute_python_file("model_testing.py", random_sentence) # use model to predict sentence and guess the language
            if predicted_language == random_sentence_and_language[1]: #compare them
                correct_count += 1
                total_count += 1
        acc_stats = correct_count / total_count if total_count > 0 else 0
        accuracy[number] = acc_stats #compute accuracy

    print(accuracy) #print and return the accuracy
    return accuracy
#-------------------------------------------------------------

def extract_random_word_and_language(number):
    languages_dict = load_training_data(size = MAX_WORD_LIST_SIZE, weighted=False)
    extract_language = random.choice(list(languages_dict.keys()))
    extract_words = []
    for _ in range(number):
        extract_words.append(random.choice(languages_dict[extract_language]))

    return (extract_words,extract_language) # store randomword and actual language to return

#-------------------------------------------------------------
def execute_python_file(file_path, random_sentence):
    try:
        # Construct the com
        # mand to execute the script
        command = 'python' + file_path + str(random_sentence)
        
        # Execute the command and capture the result
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing script '{file_path}': {e}")
        return None

#-------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("please input correct number of arguments")
        sys.exit()

    n_times = int(sys.argv[1]) #put in arguments for sys.args how many times to run,
    test_accuracy(n_times)
    print("hello")


if __name__ == '__main__':
    main()