#  IMPORT PACKAGES
import evaluate
import io
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_metric, DatasetDict
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

#select server
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# IMPORT & SET SOME PARAMETERS
direc = os.getcwd()
d = pd.read_csv(direc + '/CPSTfulldataset2.csv')

study_name = 'hp-testsearch-roberta-seed40'
model_name = "roberta-base"  # a multilingual transformer model
#prefix = "A creative solution for the situation: " # we'll use prefix/conn to construct inputs to the model
#suffix = "is: " # we'll use prefix/conn to construct inputs to the model

scaler = StandardScaler()
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # setting for whether to use gpu or cpu
np.random.seed(40) # sets a randomization seed for reproducibility


# SET UP DATASET
d['inputs'] = d['Solutions']
d['text'] = d['inputs']
d['label'] = d['FacScoresO']

d_input = d.filter(['text','label', 'set'], axis = 1)
dataset = Dataset.from_pandas(d_input, preserve_index = False) # Turns pandas data into huggingface/pytorch dataset


#  CREATE TRAIN/TEST SPLIT
dataset = Dataset.from_pandas(d_input, preserve_index = False) # Turns pandas data into huggingface/pytorch dataset
train_val_test_dataset = DatasetDict({
    'train': dataset.filter(lambda example: example['set'] == 'training'),
    'test': dataset.filter(lambda example: example['set'] == 'test'),
    'heldout': dataset.filter(lambda example: example['set'] == 'heldout')
})

train_val_test_dataset = train_val_test_dataset.remove_columns('set')

print(train_val_test_dataset) # show the dataset dictionary
print(train_val_test_dataset['train'].features)


# SET UP MODEL & TOKENIZER
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1) # TONS of settings in the model call, but labels = 1
tokenizer = AutoTokenizer.from_pretrained(model_name) # ...some settings in the tokenizer call

#  DEFINE WRAPPER TOKENIZER FUNCTION (FOR BATCH TRAINING)
def tokenize_function(examples):
  return tokenizer(examples['text'], truncation = True, padding = 'max_length')

tokenized_datasets = train_val_test_dataset.map(tokenize_function, batched = True) # applies wrapper to our dataset


#  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
def compute_metrics(eval_preds):
  predictions, references = eval_preds
  mse_metric = evaluate.load("mse")
  mse = mse_metric.compute(predictions = predictions, references = references)
  return mse



# RETRAIN

training_args = TrainingArguments(         
    output_dir= (direc), 
    learning_rate=0.00005,
    num_train_epochs=116,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    disable_tqdm=True,
    load_best_model_at_end=False,
    save_strategy = 'no',
    evaluation_strategy = 'no',
    eval_steps = 500,
    save_total_limit = 1)     


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer)

result = trainer.train() 
trainer.evaluate()



# EVALUATION

prediction = trainer.predict(tokenized_datasets['test'])
train_prediction = trainer.predict(tokenized_datasets['train'])
heldout_prediction = trainer.predict(tokenized_datasets['heldout'])


test_data = {'text':tokenized_datasets['test']['text'],'label':tokenized_datasets['test']['label'],'prediction':np.squeeze(prediction.predictions)}
train_data = {'text':tokenized_datasets['train']['text'],'label':tokenized_datasets['train']['label'],'prediction':np.squeeze(train_prediction.predictions)}
heldout_data = {'text':tokenized_datasets['heldout']['text'],'label':tokenized_datasets['heldout']['label'],'prediction':np.squeeze(heldout_prediction.predictions)}


dataset_test_df = pd.DataFrame(test_data)
dataset_train_df = pd.DataFrame(train_data)
dataset_heldout_df = pd.DataFrame(heldout_data)


dataset_test_df.to_csv('PredictedTestSet.csv')
dataset_train_df.to_csv('PredictedTrainSet.csv')
dataset_heldout_df.to_csv('PredictedHeldoutSet.csv')



