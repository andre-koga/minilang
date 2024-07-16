# used for checking the performance of the model on the test data
# and storing the results in a file.

import torch
from IO import load_training_data, load_model, store_model, MAX_WORD_LIST_SIZE
from language_code import get_language_name
from model_training import TransformerClassifier
from transformers import BertTokenizer
from prepare_dataset import prepare_dataset

def PredictLanguage(string, words_size=MAX_WORD_LIST_SIZE, path='trans.pkl', alter_base_path=False):
    # set alter_base_path to False to use the base path as is.
    # the directory is hardcoded on the IO.py file.
    full_path = f'{words_size}_{path}' if alter_base_path else path
    transformer_model = load_model(file_name=full_path)
    
    if transformer_model is None:
        print(f'\nModel file is missing or empty. Training a new model using the arguments: words_size={words_size}.')
        print(f'It will be stored at {full_path}.')
        
        train_df, test_df, label_mapping = prepare_dataset()
        
        transformer_model = TransformerClassifier(label_mapping)
        transformer_model.convert_to_torch(train_df, test_df)
        transformer_model.train()
        
        store_model(transformer_model, file_base_name=full_path)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    predicted_label = predictions.item()
    for lang, label in label_mapping.items():
        if label == predicted_label:
            return lang

word = 'avoir'
predicted_language = predict_language(word)
print(f'The predicted language for the word "{word}" is: {predicted_language}')