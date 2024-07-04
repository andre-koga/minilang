# used for loading and storing different models and different data sources.

from wordfreq import available_languages, get_frequency_dict
import os, json
import dill as pickle

# -----------------------------------------------------------------

# DO NOT CHANGE!
WORD_LIST_TYPE = 'best'
MAX_WORD_LIST_SIZE = 20000 # twenty thousand
WORDS_DIRECTORY = 'word_dicts'
MODEL_DIRECTORY = 'models'
MODEL_BASE_PATH = 'nb.pkl' # nb stands for Naive Bayes

# -----------------------------------------------------------------

def save_word_dict(lang, words):
    """
    Save the word list to a local file.
    
    Parameters:
    - lang: the language code
    - words: the list of words
    
    Returns: None
    """
    os.makedirs(WORDS_DIRECTORY, exist_ok=True)
    file_path = os.path.join(WORDS_DIRECTORY, f'{lang}.json')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(words, f)

def load_word_dict(lang):
    """
    Load the word list from a local file, if it exists.
    
    Parameters:
    - lang: the language code
    
    Returns: the list of words, or None if the file does not exist
    """
    file_path = os.path.join(WORDS_DIRECTORY, f'{lang}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    return None

def load_training_data(size = MAX_WORD_LIST_SIZE, weighted=False):
    """
    Return the data for all available languages.
    
    Parameters:
    - size: the number of words to use for training
    - weighted: whether to weight the words based on frequency
    
    Returns: a dictionary of language codes to lists of words
    """
    if size > MAX_WORD_LIST_SIZE:
        print(f'The maximum number of words is {MAX_WORD_LIST_SIZE}. Using {MAX_WORD_LIST_SIZE} instead.')
        size = MAX_WORD_LIST_SIZE
    
    data = {}
    langs = available_languages(WORD_LIST_TYPE)
    
    for lang in langs.keys():
        # Try to load from local file first
        chosen_words = load_word_dict(lang)
        
        
        if chosen_words is None:
            # If not available locally, download and save
            # TO DO: I think the weighted one could be improved. It is very slow.
            print(f'Word list for {lang} not found locally. Downloading...')

            chosen_words = get_frequency_dict(lang, WORD_LIST_TYPE)
            # limit allWords by the MAX_WORD_LIST_SIZE
            words_to_save = dict(sorted(chosen_words.items(), key=lambda item: item[1], reverse=True)[:MAX_WORD_LIST_SIZE])
            save_word_dict(lang, words_to_save)

            chosen_words = sorted(words_to_save.items(), key=lambda item: item[1], reverse=True)[:size]
        else:
            chosen_words = sorted(chosen_words.items(), key=lambda item: item[1], reverse=True)[:size]
        
        chosen_words = dict(chosen_words) if weighted else [item[0] for item in chosen_words]    
        data[lang] = chosen_words
    return data

# -----------------------------------------------------------------

def load_model(file_name = MODEL_BASE_PATH):
    full_path = os.path.join(MODEL_DIRECTORY, file_name)
    
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
    
def store_model(model, file_base_name = MODEL_BASE_PATH):
    full_path = os.path.join(MODEL_DIRECTORY, file_base_name)
    
    with open(full_path, 'wb') as file:
        pickle.dump(model, file)