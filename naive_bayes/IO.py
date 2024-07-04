# used for loading and storing different models and different data sources.

from wordfreq import top_n_list, available_languages, get_frequency_dict
import os, json
import dill as pickle

# -----------------------------------------------------------------

# DO NOT CHANGE!
WORD_LIST_TYPE = 'best'

WORDS_DIRECTORY = 'word_lists'
WEIGHTED_DIRECTORY = 'weighted_word_lists'
MODEL_DIRECTORY = 'models'
MODEL_BASE_PATH = 'nb.pkl' # nb stands for Naive Bayes

# -----------------------------------------------------------------

def save_word_list(lang, words, weighted=False):
    """
    Save the word list to a local file.
    
    Parameters:
    - lang: the language code
    - words: the list of words
    
    Returns: None
    """
    dir = WEIGHTED_DIRECTORY if weighted else WORDS_DIRECTORY
    
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, f'{lang}_{weighted}.json')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(words, f)

def load_word_list(lang, weighted=False):
    """
    Load the word list from a local file, if it exists.
    
    Parameters:
    - lang: the language code
    - dir: the directory to load the file from
    
    Returns: the list of words, or None if the file does not exist
    """
    dir = WEIGHTED_DIRECTORY if weighted else WORDS_DIRECTORY
    
    file_path = os.path.join(dir, f'{lang}_{weighted}.json'
                             )
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    return None

def load_training_data(size = 100000, weighted=False):
    """
    Return the data for all available languages.
    
    Returns: a dictionary of language codes to lists of words
    """
    data = {}
    langs = available_languages(WORD_LIST_TYPE)
    
    for lang in langs.keys():
        # Try to load from local file first
        words = load_word_list(lang, weighted=weighted)
        
        if words is None:
            # If not available locally, download and save
            # TO DO: I think the weighted one could be improved. It is very slow.
            
            if weighted:
                wordlist = get_frequency_dict(lang)
                words = sorted(wordlist, key=wordlist.get, reverse=True)[:size]
            else:
                words = top_n_list(lang, size, wordlist=wordlist)
            save_word_list(lang, words, weighted=weighted)
            
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