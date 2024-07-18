# Load the pre-trained BERT model and tokenizer, tokenize the dataset, and convert it to a torch Dataset

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from WordDataset import WordDataset

# -----------------------------------------------------------------
    
class TransformerClassifier:
    def __init__(self, label_mapping):
        # Load pre-trained tokenizer and model
        model_name = 'google/bert_uncased_L-2_H-128_A-2'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

    # Tokenize the dataset
    def tokenize_function(self, examples):
        return self.tokenizer(examples['word'], padding="max_length", truncation=True, max_length=128)
    
    def convert_to_torch(self, train_df, test_df):
        train_encodings = self.tokenizer(train_df['word'].tolist(), truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(test_df['word'].tolist(), truncation=True, padding=True, max_length=128)
        self.train_dataset = WordDataset(train_encodings, train_df['label'].tolist())
        self.test_dataset = WordDataset(test_encodings, test_df['label'].tolist())

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir='./logs',  # Directory for storing logs
            logging_steps=10,  # Log every X updates steps.
            save_strategy="epoch",  # Save checkpoint at the end of each epoch
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()