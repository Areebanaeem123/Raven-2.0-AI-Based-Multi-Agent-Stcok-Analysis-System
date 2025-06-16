import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import kagglehub
import kagglehub.datasets
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 
#loading the pretrained model 
pretrained_model = "distilbert-base-uncased-finetuned-sst-2-english" #loading the pretrained model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)#making tokens of it using transformers 
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)#loading the model
#loading and preparing the dataset 
#filepath = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
filepath_csv = "all-data.csv"
def load_data(filepath_csv):
    return pd.read_csv(
        filepath_csv,
        names = ['sentiment', 'news'],
        encoding='ISO-8859-1', 
        quoting=csv.QUOTE_MINIMAL,
        )
loaded_dataset = load_data(filepath_csv)
#claning and mapping sentiment 
loaded_dataset['sentiment']= loaded_dataset['sentiment'].str.strip().map({
    'positive': 'positive',
    'neutral':'neutral',
    'negative':'negative'
})    
print(loaded_dataset.head())
loaded_dataset = load_data(filepath_csv)
#now tokenizing the text into input IDs
dataset = Dataset.from_pandas(loaded_dataset[['news','sentiment']])
def tokenize_function(examples):
    return tokenizer(examples['news'], truncation=True, padding=True)
tokenized_dataset=dataset.map(tokenize_function , batched= True )
#no splkitting the dataset into testing and training \
dataset = dataset.train_test_split(test_size = 0.2, seed = 42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
#now fine tuning the model 
training_args= TrainingArguments(
    output_dir="./fine-tuned-model",
    do_eval = True ,
    learning_rate =2e-5,#how quickly the model is updating the weights 
    per_device_train_batch_size=16,
    per_device_eval_batch_size = 16,
    num_train_epochs=3,
    weight_decay = 0.01,
)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=3)#oit means three clasess positive , neutral and negative 
trainer = Trainer (
    model = model,
    arguments = training_args,
    train_dataset= train_dataset ,
    eval_dataset = test_dataset,
    tokenizer = tokenizer,
)
trainer.train()
#now evaluating the accuracy 
def calculate_accuracy(eval_pred):
    logits , labels = eval_pred
    predictions = np.argmax(logits , axis = -1)
    model_accuracy = accuracy_score(labels, predictions)
    return {"accuracy of model: ", model_accuracy}
trainer = Trainer(
    model = model,
    arguments = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset ,
    tokenizer = tokenizer,
    compute_metrics = calculate_accuracy,
)
trainer.train()
#final evaluation on the test set 
metrics = trainer.evaluate()
print("the metrics are as follows:", metrics)
#saving the fine tuned model 







