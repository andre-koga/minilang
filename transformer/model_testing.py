import torch
from IO import load_training_data, load_model, store_model, MAX_WORD_LIST_SIZE
from language_code import get_language_name
from model_training import TranformerClassifier
from transformers import BertTokenizer
from prepare_dataset import label_mapping

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def PredictLanguage(string, words_size=MAX_WORD_LIST_SIZE, n_grams=(1, 2, 3), weighted=False, path='trans.pkl', alter_base_path=False):
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