import os
import zipfile
import transformers
import numpy as np
import pandas as pd
from datasets import Dataset,load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from accelerate import PartialState

os.environ['KAGGLE_CONFIG_DIR'] = 'EmoSense'
os.system("kaggle datasets download -d karthikrathod/emosense-models")

# Specify the path to the zip file
zip_path = 'emosense-models.zip'

# Specify the directory where you want to extract the contents of the zip file
extract_dir = 'emosense-models'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents of the zip file to the specified directory
    zip_ref.extractall(extract_dir)

def get_path(x):
  
  # Specify the filename or file path
  filename = x # Change this to your desired file name or path

  # Get the absolute path of the file
  file_path = os.path.abspath(filename)

  # Print the file path
  return file_path

model_V = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_V'))
tokenizer_V = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_V'))
trainer_V = Trainer(model=model_V)

model_A = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/model_A'))
tokenizer_A = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_A'))
trainer_A = Trainer(model=model_A)

model_D = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_D'))
tokenizer_D = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_D'))
trainer_D = Trainer(model=model_D)


def predict_vad(sentence:str):

  dataset = Dataset.from_pandas(pd.DataFrame({'text':[sentence]}),preserve_index=False) # converting input text to dataset

  #====================================MODEL_V===============================================
  # model_V = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_V'))
  # tokenizer_V = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_V'))
  # trainer_V = Trainer(model=model_V)
  def tokenize_function_V(examples):
      return tokenizer_V(examples["text"], truncation=True) # Tokenization function to tokenize the input text
  def pipeline_prediction_V(dataset):
      
      tokenized_datasets_V = dataset.map(tokenize_function_V)
      raw_pred, _, _ = trainer_V.predict(tokenized_datasets_V) # predicting the specific varaible using respective model and tokenizer
      return(raw_pred[0][0])
 #=======================================MODEL_A===============================================
  # model_A = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/model_A'))
  # tokenizer_A = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_A'))
  # trainer_A = Trainer(model=model_A)
  def tokenize_function_A(examples):
      return tokenizer_A(examples["text"], truncation=True) 
  def pipeline_prediction_A(dataset):
      
      tokenized_datasets_A = dataset.map(tokenize_function_A)
      raw_pred, A, B = trainer_A.predict(tokenized_datasets_A) 
      return(raw_pred[0][0])
  #======================================MODEL_D================================================
  # model_D = AutoModelForSequenceClassification.from_pretrained(get_path('emosense-models/EMO_MODELS/MODEL_D'))
  # tokenizer_D = AutoTokenizer.from_pretrained(get_path('emosense-models/EMO_MODELS/tokenizer_D'))
  # trainer_A = Trainer(model=model_A)

  def tokenize_function_D(examples):
      return tokenizer_D(examples["text"], truncation=True) 
  def pipeline_prediction_D(dataset):
      

      tokenized_datasets_D = dataset.map(tokenize_function_D)
      raw_pred, _, _ = trainer_D.predict(tokenized_datasets_D) 
      return(raw_pred[0][0])

  return pipeline_prediction_V(dataset),pipeline_prediction_A(dataset),pipeline_prediction_D(dataset)
