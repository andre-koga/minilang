# This file contains the language code and their respective language name.

LANGUAGE_CODE = {
    'ar': 'Arabic',
    'bn': 'Bangla',
    'bs': 'Bosnian',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ms': 'Malay',
    'nb': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sr': 'Serbian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'fil': 'Tagalog',
    'ta': 'Tamil',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese'
}

non_logographic_langs = [
    'ca', 'cs', 'da', 'de', 'en', 'es', 'fi', 'fil', 'fr', 'hu', 'id', 'is', 'it', 'lt', 'lv', 'ms', 'nb', 'nl', 'pl', 'pt', 'ro', 'sh', 'sk', 'sl', 'sv', 'tr'
]

def get_language_name(code):
    """
    Get the name of the language corresponding to the given code.
    
    Parameters:
    - code: the language code
    
    Returns: the name of the language
    """
    return LANGUAGE_CODE[code] if code in LANGUAGE_CODE else None


# Language    Code    #  Large?   WP    Subs  News  Books Web   Twit. Redd. Misc.
# ──────────────────────────────┼────────────────────────────────────────────────
# Arabic      ar      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Bangla      bn      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Bosnian     bs [1]  3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Bulgarian   bg      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Catalan     ca      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Chinese     zh [3]  7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   -     Jieba
# Croatian    hr [1]  3         │ Yes   Yes   -     -     -     Yes   -     -
# Czech       cs      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Danish      da      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Dutch       nl      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# English     en      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Finnish     fi      6  Yes    │ Yes   Yes   Yes   -     Yes   Yes   Yes   -
# French      fr      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# German      de      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Greek       el      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Hebrew      he      5  Yes    │ Yes   Yes   -     Yes   Yes   Yes   -     -
# Hindi       hi      4  Yes    │ Yes   -     -     -     Yes   Yes   Yes   -
# Hungarian   hu      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Icelandic   is      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Indonesian  id      3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Italian     it      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Japanese    ja      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Korean      ko      4  -      │ Yes   Yes   -     -     -     Yes   Yes   -
# Latvian     lv      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Lithuanian  lt      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Macedonian  mk      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Malay       ms      3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Norwegian   nb [2]  5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Persian     fa      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Polish      pl      6  Yes    │ Yes   Yes   Yes   -     Yes   Yes   Yes   -
# Portuguese  pt      5  Yes    │ Yes   Yes   Yes   -     Yes   Yes   -     -
# Romanian    ro      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Russian     ru      5  Yes    │ Yes   Yes   Yes   Yes   -     Yes   -     -
# Slovak      sk      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Slovenian   sl      3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Serbian     sr [1]  3  -      │ Yes   Yes   -     -     -     Yes   -     -
# Spanish     es      7  Yes    │ Yes   Yes   Yes   Yes   Yes   Yes   Yes   -
# Swedish     sv      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Tagalog     fil     3  -      │ Yes   Yes   -     -     Yes   -     -     -
# Tamil       ta      3  -      │ Yes   -     -     -     Yes   Yes   -     -
# Turkish     tr      4  -      │ Yes   Yes   -     -     Yes   Yes   -     -
# Ukrainian   uk      5  Yes    │ Yes   Yes   -     -     Yes   Yes   Yes   -
# Urdu        ur      3  -      │ Yes   -     -     -     Yes   Yes   -     -
# Vietnamese  vi      3  -      │ Yes   Yes   -     -     Yes   -     -     -