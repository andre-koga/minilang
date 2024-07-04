import os
import json
import random

def test_accuracy(test_data, model):
    """
    Tests the accuracy of the model on the given test data.

    Parameters:
    - test_data (dict): A dictionary where keys are language labels and values are lists of strings in that language.
    - model (NaiveBayesClassifier): The trained NaiveBayesClassifier model.

    Returns:
    - float: The accuracy of the model on the test data.
    """
    total_count = 0
    correct_count = 0
    test_number = sys.argv[1]
    for _ in range(test_number):
        test_data = extract_random_word_and_language()
        predicted_language = model.predict(string)
        if predicted_language == test_data:
            correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy



WORD_LIST_DIRECTORY = 'word_list'

def extract_random_word_and_language():
    """
    Extract a random word and its corresponding language from JSON files in a directory.

    Returns:
    - tuple: A tuple containing the randomly selected word and its language code.
    """
    # List all JSON files in the directory

# Example usage
if __name__ == "__main__":
    word, language = extract_random_word_and_language()
    if word and language:
        print(f"Randomly selected word: '{word}' from language: '{language}'")
    else:
        print("Failed to extract a random word and its language.")