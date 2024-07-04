# used for loading and storing different models and different data sources.

from wordfreq import top_n_list, available_languages
import os, json
import dill as pickle

# -----------------------------------------------------------------

WORDS_DIRECTORY = 'word_lists'
MODEL_DIRECTORY = 'models'
MODEL_BASE_PATH = 'nb.pkl' # nb stands for Naive Bayes

# -----------------------------------------------------------------

def save_word_list(lang, words, dir = WORDS_DIRECTORY):
    """
    Save the word list to a local file.
    
    Parameters:
    - lang: the language code
    - words: the list of words
    - dir: the directory to save the file in
    
    Returns: None
    """
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, f'{lang}.json')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(words, f)

def load_word_list(lang, dir = WORDS_DIRECTORY):
    """
    Load the word list from a local file, if it exists.
    
    Parameters:
    - lang: the language code
    - dir: the directory to load the file from
    
    Returns: the list of words, or None if the file does not exist
    """
    file_path = os.path.join(dir, f'{lang}.json'
                             )
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    return None

def load_training_data(size = 100000, wordlist = 'best'):
    """
    Return the data for all available languages.
    
    Returns: a dictionary of language codes to lists of words
    """
    data = {}
    langs = available_languages(wordlist='best')
    
    for lang in langs.keys():
        # Try to load from local file first
        words = load_word_list(lang)
        
        if words is None:
            # If not available locally, download and save
            words = top_n_list(lang, size, wordlist=wordlist)
            save_word_list(lang, words)
            
        data[lang] = words
        
    return data

# -----------------------------------------------------------------

def load_model(file_name = MODEL_BASE_PATH, dir = MODEL_DIRECTORY):
    full_path = os.path.join(dir, file_name)
    
    if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
        # File does not exist or is empty
        return None
    try:
        with open(full_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except (EOFError, FileNotFoundError) as e:
        print(f"Error loading model: {e}")
        return None
    
def store_model(model, file_base_name = MODEL_BASE_PATH, dir = MODEL_DIRECTORY):
    full_path = os.path.join(dir, file_base_name)
    
    with open(full_path, 'wb') as file:
        pickle.dump(model, file)