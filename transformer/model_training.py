# Load the pre-trained BERT model and tokenizer, tokenize the dataset, and convert it to a torch Dataset

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from WordDataset import WordDataset
from prepare_dataset import train_df, test_df, label_mapping

# -----------------------------------------------------------------


    
class TransformerClassifier:
    def __init__(self):
        # Load pre-trained tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))

    # Tokenize the dataset
    def tokenize_function(self, examples):
        return self.tokenizer(examples['word'], padding="max_length", truncation=True)

    def prepare_dataset(self):
        train_encodings = self.tokenizer(train_df['word'].tolist(), truncation=True, padding=True)
        test_encodings = self.tokenizer(test_df['word'].tolist(), truncation=True, padding=True)
        self.train_dataset = WordDataset(train_encodings, train_df['label'].tolist())
        self.test_dataset = WordDataset(test_encodings, test_df['label'].tolist())

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
        )

        trainer.train()