# used for checking the performance of the model on the test data
# and storing the results in a file.

import sys
import torch
import IO, Lang
from transformers import BertTokenizer
from model_training import TransformerClassifier
from prepare_dataset import prepare_dataset

def PredictLanguage(string, words_size=IO.MAX_WORD_LIST_SIZE, path='trans.pkl', alter_base_path=False):
    # set alter_base_path to False to use the base path as is.
    # the directory is hardcoded on the IO.py file.
    full_path = f'{words_size}_{path}' if alter_base_path else path
    transformer_model = IO.load_model(file_name=full_path)
    
    if transformer_model is None:
        print(f'\nModel file is missing or empty. Training a new model using the arguments: words_size={words_size}.')
        print(f'It will be stored at {full_path}.')
        
        train_df, test_df, label_mapping = prepare_dataset()
        
        transformer_model = TransformerClassifier(label_mapping)
        transformer_model.convert_to_torch(train_df, test_df)
        transformer_model.train()
        
        IO.store_model(transformer_model, file_base_name=full_path)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(string, return_tensors="pt", truncation=True, padding=True)
    outputs = transformer_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    predicted_label = predictions.item()
    for lang, label in label_mapping.items():
        if label == predicted_label:
            return lang

# -----------------------------------------------------------------

def main():
    if len(sys.argv) == 2:
        # order of arguments:
        # 1. (string) string to predict the language of

        # an example of how to run the script:
        # python model_testing.py "this is a test string"
        
        # IMPORTANT! this by default uses the top MAX_WORD_LIST_SIZE words and n-grams (1, 2, 3) and weighted=False
        
        string = sys.argv[1]
        
        lang = PredictLanguage(string)
        
    print(f'The predicted language for the string "{string}" is: {Lang.get_language_name(lang)}\n')
    print(f'Bear in mind that lowercase and uppercase may affect the prediction.')
    
if __name__ == '__main__':
    main()