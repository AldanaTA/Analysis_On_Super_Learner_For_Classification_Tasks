from transformers import AutoModelForSequenceClassification,RobertaTokenizer, Trainer, TrainingArguments 
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import numpy as np
import os


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.label = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
       
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation = True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding =  'max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
   


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.label[index], dtype=torch.float)
        }


def encode_Lables(label):
    temp = np.zeros(5)
    temp[label-1] = 1
    return temp


    
def split_data(filename,samp_size):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    path = parent_dir +"/model_data/" + filename
  
    df = pd.read_csv(path)
 #Sampling data
    if type(samp_size) is float:
        if samp_size <=0:
            samp_size = .1 
        n = int(len(df) * samp_size)
        if n > len(df):
            print('sample size must be less than or equal to input size set sample to 4000')
            n = 4000
    else:
        if samp_size > 1 and len(df) >= samp_size:
            n = samp_size
        else:
            print('sample size must be less than or equal to input size set sample to 4000')
            n = 4000
        
    df = df.sample(n = n)
    df.reset_index(inplace=True, drop= True)

    #Renaming for Model
    df.rename(columns={"Rating":"label"},inplace=True)
    df.rename(columns={"Review_text":"text"},inplace=True)
    df["label"] = df["label"].astype(int).apply(encode_Lables)
    train_data,test_data = train_test_split(df,test_size=.20,random_state=64)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    max_length = int(df["text"].str.split().str.len().mean())
    return train_data,test_data,max_length



def compute_metrics(pred):
    
    label = torch.argmax(torch.tensor(pred.label_ids),dim=1).numpy()
    preds = torch.argmax(torch.tensor(pred.predictions),dim=1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(label, preds, average='macro',zero_division=0.0)
    acc = accuracy_score(label, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_model(filename,samp_size,lr,epochs,w_decay):
    train,test,max_length= split_data(filename,samp_size)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print(max_length)
    training_set = SentimentData(train, tokenizer, max_len=50)
    testing_set = SentimentData(test, tokenizer, max_len=50)

    training_args = TrainingArguments(

    learning_rate=lr,
    fp16=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=epochs,
    weight_decay=w_decay,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to= 'none'
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer.train()
    return trainer





