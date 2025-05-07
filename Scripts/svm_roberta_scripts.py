from transformers import RobertaTokenizer,RobertaForSequenceClassification,pipeline,TextClassificationPipeline
import pandas as pd
import torch
from sklearn.model_selection import KFold,cross_validate
from torch.utils.data import Dataset
from sklearn.metrics import make_scorer,accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import os
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import tuning_script


class Pipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        return best_class.squeeze().numpy()
class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
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
        }
    
def read_data(filename,samp_size):
     #reading Data
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
    df.rename(columns={"Review_text":"text"},inplace=True)
    df["Rating"] = df["Rating"].apply(encode_Test_Lables)
    return df

def encode_Test_Lables(label):
    return int(label-1)


filename = "data_set_1.csv"
samp_size = 5000
def run_model(filename,samp_size,svm_C,meta_C,model_path):
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    data = read_data(filename,samp_size)
    text = data["text"].to_list()

    #creating pipeline and predicting with roberta model
    classifier = Pipeline(model = model,tokenizer = tokenizer)
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    pred = classifier(text,**tokenizer_kwargs)

    roberta_pred = pd.DataFrame(pred)
    roberta_pred = roberta_pred.to_numpy()

    #Convert text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_df=.95,min_df=.0125)
    svm_x = vectorizer.fit_transform(data["text"])

    # create model instance
    classifier_svc = SVC(C=svm_C,class_weight='balanced',random_state = 50,probability=True)

    # fit model
    classifier_svc.fit(svm_x,data["Rating"])

    #predict
    svm_results = classifier_svc.predict_proba(svm_x)

    #Create meta data from roberta and svm predictions should be a num_models * (predictions * classes)
    meta_data = np.concatenate((roberta_pred, svm_results), axis=1)

    meta_svm = SVC(C=meta_C,class_weight='balanced',random_state = 50)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score,average="macro",zero_division=0.0),
        'recall': make_scorer(recall_score,average="macro",zero_division=0.0),
        'f1_score': make_scorer(f1_score,average="macro",zero_division=0.0)
    }      
    cv_results = cross_validate(meta_svm, meta_data, data["Rating"], cv=kf, scoring=scoring)
    
    print("Model Accuracy: {0:.2%} Precision: {1:.4f} Recall: {2:.4f} F1 Score: {3:.4f}".format(np.mean(cv_results['test_accuracy']),np.mean(cv_results['test_f1_score']),np.mean(cv_results['test_precision']),np.mean(cv_results['test_recall'])))


def run_experiments(filename,samp_size,svm_C,meta_C,model_path):
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    data = read_data(filename,samp_size)
    text = data["text"].to_list()

    #creating pipeline and predicting with roberta model
    classifier = Pipeline(model = model,tokenizer = tokenizer)
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    pred = classifier(text,**tokenizer_kwargs)

    roberta_pred = pd.DataFrame(pred)
    roberta_pred = roberta_pred.to_numpy()

    #Convert text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_df=.95,min_df=.0125)
    svm_x = vectorizer.fit_transform(data["text"])

    # create model instance
    classifier_svc = SVC(C=svm_C,class_weight='balanced',random_state = 50,probability=True)

    # fit model
    classifier_svc.fit(svm_x,data["Rating"])

    #predict
    svm_results = classifier_svc.predict_proba(svm_x)

    #Create meta data from roberta and svm predictions should be a num_models * (predictions * classes)
    meta_data = np.concatenate((roberta_pred, svm_results), axis=1)

    meta_svm = SVC(C=.001,class_weight='balanced',random_state = 50)
    
    paramaters = {
        'C' : meta_C
    }
    params = tuning_script.grid_search(meta_svm,paramaters,meta_data,data["Rating"])
    return params


