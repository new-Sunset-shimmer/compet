# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
cuda_value = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_value
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup


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
df = pd.read_csv('~/data/train_data.csv')
df['label'].unique()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['label'])
df1 = pd.read_csv('~/data/test_data.csv')

sentences = df1.Title.values

#check distribution of data based on labels

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 
# In the original paper, the authors used a length of 512.
MAX_LEN = 88

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased',do_lower_case=True)
input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=24).to(device)
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
model = torch.load("model0.09141193595342068.pt")
# os.makedirs('out', exist_ok=True)
with torch.no_grad(), open('/home2/yangcw/submission'+f'{cuda_value}'+'.csv', 'w') as fd:
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