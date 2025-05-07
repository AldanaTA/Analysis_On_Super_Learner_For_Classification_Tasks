
from sklearn.svm import SVC
from sklearn.metrics import make_scorer,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold,cross_validate
import pandas as pd
import numpy as np
import os
import tuning_script


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
    return df

def encode_Test_Lables(label):
    return int(label-1)

def run_model(filename,samp_size,l2,max_df,min_df):
    df = read_data(filename,samp_size)
    vectorizer = TfidfVectorizer(max_df= max_df,min_df= min_df)
    x = vectorizer.fit_transform(df["Review_text"])
    df["Rating"] = df["Rating"].apply(encode_Test_Lables)
    classifier_svc = SVC(C=l2,class_weight='balanced',random_state = 50)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score,average="macro",zero_division=0.0),
    'recall': make_scorer(recall_score,average="macro",zero_division=0.0),
    'f1_score': make_scorer(f1_score,average="macro",zero_division=0.0)
    }
    
    cv_results = cross_validate(classifier_svc, x, df["Rating"].astype(int), cv=kf, scoring=scoring)

    print("Model Accuracy: {0:.2%} Precision: {1:.4f} Recall: {2:.4f} F1 Score: {3:.4f}".format(np.mean(cv_results['test_accuracy']),np.mean(cv_results['test_f1_score']),np.mean(cv_results['test_precision']),np.mean(cv_results['test_recall'])))



def run_experiments(filename,samp_size,l2,max_df,min_df):
    
    df = read_data(filename,samp_size)
    vectorizer = TfidfVectorizer(max_df= max_df,min_df= min_df)
    x = vectorizer.fit_transform(df["Review_text"])
    df["Rating"] = df["Rating"].apply(encode_Test_Lables)
    classifier_svc = SVC(C=l2,class_weight='balanced',random_state = 50)
    paramaters = {
        'C' : l2
    }
    params = tuning_script.grid_search(classifier_svc,paramaters,x,df["Rating"])
    return params
    
    

'''
l2 = [.852,1.2,1.4,1.6]
max_df = .97
min_df = .001
runExperiments('data_set_2.csv',.5,l2,max_df,min_df)
'''

