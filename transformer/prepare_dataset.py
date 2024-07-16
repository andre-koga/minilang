# Prepare a dataset for training a transformer model

import pandas as pd
from sklearn.model_selection import train_test_split
from .. import IO

# -----------------------------------------------------------------

def prepare_dataset():
    data = IO.load_training_data()
    
    # Convert to a DataFrame
    df = pd.DataFrame([(word, lang) for lang, words in data.items() for word in words], columns=['word', 'language'])
    
    # Encode labels
    label_mapping = {lang: i for i, lang in enumerate(df['language'].unique())}
    df['label'] = df['language'].map(label_mapping)
    
    # Split into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df, label_mapping