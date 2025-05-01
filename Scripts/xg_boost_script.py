from xgboost import XGBClassifier
import pandas as pd
import torch
from sklearn.model_selection import KFold,cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer,accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
import tuning_script

def read_data(filename,samp_size):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    path = parent_dir +"/model_data/" + filename
    #Sampling data
    if samp_size > 1:
        samp_size = 1
    if samp_size <=0:
      samp_size = .1

    df = pd.read_csv(path)
    if int(len(df) * samp_size) <= 5000:
        df = df.sample(n=int(len(df) * samp_size),random_state= 33)
    else:
        df = df.sample(n=5000)  
   
    df.reset_index(inplace=True,drop=True)

    return df

def encode_Test_Lables(label):
    return int(label-1)

def run_model(filename,samp_size,max_df,min_df,l2,lr,eta):
    df = read_data(filename,samp_size)

    device = "cpu"
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda'

    # Convert text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_df=max_df,min_df=min_df)
    x = vectorizer.fit_transform(df["Review_text"])
    y = df["Rating"].apply(encode_Test_Lables)

    # create model instance
    bst = XGBClassifier(n_estimators = 20,reg_lambda = l2, eta = eta, learning_rate = lr, booster = 'gblinear', objective='multi:softmax',device = device, random_state = 50)

    #Set folds and Scoring
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score,average="micro",zero_division=0.0),
    'recall': make_scorer(recall_score,average="micro",zero_division=0.0),
    'f1_score': make_scorer(f1_score,average="micro",zero_division=0.0)
    }
    
    #run model
    cv_results = cross_validate(bst, x, y, cv=kf, scoring=scoring)

    #print results
    print("Model Accuracy: {0:.2%} Precision: {1:.4f} Recall: {2:.4f} F1 Score: {3:.4f}".format(np.mean(cv_results['test_accuracy']),np.mean(cv_results['test_f1_score']),np.mean(cv_results['test_precision']),np.mean(cv_results['test_recall'])))



def run_Experiment(filename,samp_size,max_df,min_df,l2,lr,eta):
    df = read_data(filename,samp_size)

    device = "cpu"
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda'

    # Convert text data into numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_df = max_df,min_df=min_df)
    x = vectorizer.fit_transform(df["Review_text"])
    y = df["Rating"].apply(encode_Test_Lables)

    # create model instance
    bst = XGBClassifier(n_estimators = 20,booster='gblinear',objective='multi:softmax',device = device, random_state = 50)
    paramaters = {
        'reg_lambda': l2,
        'learning_rate': lr,
        'eta': eta
    }

    params = tuning_script.grid_search(bst,paramaters,x,y)
    return params

'''
lr = [.0001,.0005,.001]
estimators = [5,10,15,20]
l2 = [.552,.652,.753,.852]
max_df = .97
min_df = .001
run_Experiment('data_set_2.csv',.5,estimators,lr,l2,max_df,min_df)
'''

