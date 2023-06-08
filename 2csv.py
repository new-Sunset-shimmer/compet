# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
cuda_value = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_value
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
# from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
# import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel,AdamW,get_linear_schedule_with_warmup,AutoModelForSequenceClassification

import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
# Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import accuracy_score,matthews_corrcoef

from tqdm import tqdm, trange,tnrange,tqdm_notebook
import random
import os
import io
# identify and specify the GPU as the device, later in training loop we will load data into device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

SEED = 19

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda")
df = pd.read_csv('~/data/Real_train_book_filter2_data_ver2.csv')
df1 = pd.read_csv('~/data/Real_val_book_filter2_data_ver2.csv')
df['label'].unique()
df1['label'].unique()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['label_enc'] = labelencoder.fit_transform(df['label'])
df[['label','label_enc']].drop_duplicates(keep='first')
df1['label_enc'] = labelencoder.fit_transform(df1['label'])
df1[['label','label_enc']].drop_duplicates(keep='first')
## create label and sentence list
sentences = df.Title.values
sentences1 = df1.Title.values
#check distribution of data based on labels
print("Distribution of data based on labels: ",df.label_enc.value_counts())
print("Distribution of data based on labels: ",df1.label_enc.value_counts())
# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
# In the original paper, the authors used a length of 512.
MAX_LEN = 88

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased',do_lower_case=True)
# input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
# labels = df.label_enc.values
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2',do_lower_case=True)
input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
labels = df.label_enc.values
input_ids1 = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences1]
labels1 = df1.label_enc.values
print("Actual sentence before tokenization: ",sentences[2])
print("Encoded Input from dataset: ",input_ids[2])

## Create attention mask
attention_masks = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

attention_masks1 = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
attention_masks1 = [[float(i>0) for i in seq] for seq in input_ids1]

# print(attention_masks[2])
# train_inputs,validation_inputs,train_labels,validation_labels = train_test_split(input_ids,labels,random_state=41,test_size=0.1)
# train_masks,validation_masks,_,_ = train_test_split(attention_masks,input_ids,random_state=41,test_size=0.1)
# convert all our data into torch tensors, required data type for our model
print("to.tensor")
train_inputs = torch.tensor(input_ids).clone().detach()
validation_inputs = torch.tensor(input_ids1).clone().detach()
train_labels = torch.tensor(labels).clone().detach()
validation_labels = torch.tensor(labels1).clone().detach()
train_masks = torch.tensor(attention_masks).clone().detach()
validation_masks = torch.tensor(attention_masks1).clone().detach()
del input_ids,input_ids1,labels,labels1,attention_masks,attention_masks1
# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 256
accumulation = 128
# batch_size = 5

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory
print("train")
train_data = TensorDataset(train_inputs,train_masks,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size// accumulation)
print("validation")
validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size// accumulation)
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
# model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=24).to(device)
print("model")
model = AutoModelForSequenceClassification.from_pretrained('intfloat/e5-large-v2', num_labels=24).to(device)
# Parameters:
lr = 2e-5
adam_epsilon = 1e-8

# Number of training epochs (authors recommend between 2 and 4)
epochs = 7

num_warmup_steps = 0
num_training_steps = len(train_dataloader)*epochs

### In Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False,no_deprecation_warning=True)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []

# Gradients gets accumulated by default
model.zero_grad()
maxer = 0
# tnrange is a tqdm wrapper around the normal python range
for _ in tqdm(range(epochs)):
  print("<" + "="*22 + F" Epoch {_} "+ "="*22 + ">")
  # Calculate total loss for this epoch
  batch_loss = 0
  running_loss = 0
  for step, batch in enumerate(train_dataloader):
    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Forward pass
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    # loss = outputs.loss.type(torch.FloatTensor)
    loss = outputs.loss
    # Backward pass
    (loss / accumulation).backward()
    running_loss += loss.item()
    
    if step % accumulation: # step % acc == 0이 아니면 다시 backward하러 돌아가게끔
        continue
    # Clip the norm of the gradients to 1.0
    # Gradient clipping is not in AdamW anymore
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    # Update learning rate schedule
    scheduler.step()

    # Clear the previous accumulated gradients
    optimizer.zero_grad()
    
    # Update tracking variables
    batch_loss += running_loss
    running_loss = 0
    
  # Calculate the average loss over the training data.
  avg_train_loss = batch_loss / len(train_dataloader)

  #store the current learning rate
  for param_group in optimizer.param_groups:
    print("\n\tCurrent Learning rate: ",param_group['lr'])
    learning_rate.append(param_group['lr'])
    
  train_loss_set.append(avg_train_loss)
  print(F'\n\tAverage Training loss: {avg_train_loss}')
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_accuracy,eval_mcc_accuracy,nb_eval_steps = 0, 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    # Move logits and labels to CPU
    logits = logits[0].to('cpu').numpy()
    label_ids = b_labels.to('cpu').numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    df_metrics=pd.DataFrame({'Epoch':epochs,'Actual_class':labels_flat,'Predicted_class':pred_flat})
    
    tmp_eval_accuracy = accuracy_score(labels_flat,pred_flat)
    tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)
    
    eval_accuracy += tmp_eval_accuracy
    eval_mcc_accuracy += tmp_eval_mcc_accuracy
    nb_eval_steps += 1
  if eval_accuracy/nb_eval_steps>maxer:
    q23 = eval_accuracy/nb_eval_steps
    torch.save(model,'2csvmodel'+f'{q23}'+'.pt')
    maxer = eval_accuracy/nb_eval_steps
    with open("/home2/yangcw/"+"2csv_model"+f"{q23}"+"txt", "w") as file:
        file.write("2csv: "+"batch_size:"+f"{batch_size}"+" accumulation:"+f"{accumulation}"+" SEED:"+f"{SEED}"+" lr:"+f"{lr}"+" adam_epsilon:"+f"{adam_epsilon}")
  print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
  print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')

df1 = pd.read_csv('~/data/test_data.csv')

sentences = df1.Title.values

#check distribution of data based on labels

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2',do_lower_case=True)
input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]

print("Actual sentence before tokenization: ",sentences[0])
print("Encoded Input from dataset: ",input_ids[0])

## Create attention mask
attention_masks = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]
print(attention_masks[2])

# convert all our data into torch tensors, required data type for our model
train_inputs = torch.tensor(input_ids).clone().detach()
train_masks = torch.tensor(attention_masks).clone().detach()

# batch_size = 5
batch_size = 1
# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs,train_masks)
train_sampler = RandomSampler(train_data)
test_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle = False)

import csv

# os.makedirs('out', exist_ok=True)
with torch.no_grad(), open('/home2/yangcw/run_submission'+f'{cuda_value}'+'.csv', 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['id', 'label'])
#     with torch.no_grad():
#             outputs = model(**input)
    rows = []
    for step, batch in enumerate(test_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.logits[0].to('cpu').numpy()
        pred_flat = logits.argmax(axis=-1).flatten()
        rows.append([step, labelencoder.inverse_transform(pred_flat)[0]])
    writer.writerows(rows)