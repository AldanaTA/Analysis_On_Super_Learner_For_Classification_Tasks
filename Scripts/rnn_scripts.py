
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold,cross_validate
from skorch import NeuralNetClassifier
from sklearn.metrics import make_scorer,accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
import numpy as np
import tuning_script


class wordVector:
    def __init__(self,vector,index = 0,key = '<pad>'):
        self.index = index
        self.key = key
        self.vector = vector

def get_embeddings(wv_objs):
    temp = []
    for objs in wv_objs.values():
        temp.append(objs.vector)
    return np.array(temp)

def encode_Test_Lables(label):
    return int(label-1)

def encode_Train_Lables(label):
    temp = np.zeros(5)
    temp[label-1] = 1
    return temp


def splitsentences(sentences):
    splits = []
    for x in range(len(sentences)):
        splits.append(sentences[x].split())
    return splits

def encodeSentence(sentence,embeds,max):
    encoding = np.zeros(max)
    for i in range(len(encoding)):
        index = 0
        if i < len(sentence):
            try:
                index = embeds[sentence[i]].index
            except:
                index = 0
        encoding[i] = index
    return encoding.astype(int)
    
def encodeSentences(data,embeds,max):
    splitted_sentences = splitsentences(data)
    encoded_Sentences = []
    for sentence in splitted_sentences:
        encoded_Sentences.append(encodeSentence(sentence,embeds,max))
    return np.array(encoded_Sentences)

def shrinkEmbeds(corpus,embeds):
    words_index = dict()
    words_index['<pad>'] = wordVector(embeds[0],0,'<pad>')
    i = 1
    for word in corpus:
        if word not in words_index:
            try:
                words_index[word] = wordVector(embeds[word],i,word)
                i += 1
            except: 
                pass 
    return words_index

def prep_data(filename,samp_size,embeds,min_df,max_df):
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
    #preping features
    vectorizer = TfidfVectorizer(min_df=min_df,max_df=max_df)
    corpus = vectorizer.fit(df["Review_text"]).get_feature_names_out()
    embeds = shrinkEmbeds(corpus,embeds)
    mean = int(df["Review_text"].str.split().str.len().mean())

    X = encodeSentences(df["Review_text"],embeds,mean)
    #del vectorizer,text_vector
    #gc.collect()

    #preping labels
    Y = df["Rating"].astype(int).apply(encode_Test_Lables)
    
    
    return X,Y,embeds

class RNNClassifier(torch.nn.Module):
    def __init__(self,embeddings='',vocabSize='',device='cpu',hidden_size = 10,num_layers = 1,):
        super(RNNClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device= device
        self.embeds = torch.nn.Embedding(vocabSize,300,_freeze=False,device=device)
        self.embeds.weight = torch.nn.Parameter(torch.from_numpy(embeddings))
        self.rnn = torch.nn.RNN(300, hidden_size,num_layers,nonlinearity="relu", batch_first=True,device=device)
        
        self.fc = torch.nn.Linear(hidden_size, 5)
        self.softMax = torch.nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        input = self.embeds(x).to(torch.float32)
        input = input.to(self.device)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size, device= self.device)
        out, _ = self.rnn(input,h0)
      
        out = out[:, -1, :]
       
        out = self.fc(out)
        
        out = self.softMax(out)
        if out.device == 'cuda':
            out = out.detach()
        return out

def run_experiments(filename,sample_size,embeddings,min_df,max_df,lr,weight_decay):
    X_data,Y_data,embeddings = prep_data(filename,sample_size,embeddings,min_df,max_df)
    embeddings = get_embeddings(embeddings)
    vocab_size = len(embeddings)

    device = "cpu"
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda'
    model = RNNClassifier(embeddings=embeddings,vocabSize=vocab_size,device=device)

    model.to(device)
    #setting paramaters
    net = NeuralNetClassifier(
    module = model,
    criterion= torch.nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    module__embeddings=embeddings,
    module__vocabSize=vocab_size,
    optimizer__weight_decay = .01,
    optimizer__lr = .001,
    batch_size = 8,
    max_epochs=5,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,

    )

    net.set_params(train_split=False, verbose=0)
    parameters = {
        'optimizer__weight_decay': weight_decay,
        'optimizer__lr': lr,
        
    }

    params = tuning_script.grid_search(net,parameters,X_data,Y_data)
    return params
    
def run_model(filename,sample_size,embeddings,min_df,max_df,lr,decay):
    X_data,Y_data,embeddings = prep_data(filename,sample_size,embeddings,min_df,max_df)
    embeddings = get_embeddings(embeddings)
    vocab_size = len(embeddings)

    device = "cpu"
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda'
    model = RNNClassifier(embeddings=embeddings,vocabSize=vocab_size,device=device)

    model.to(device)
    #setting paramaters
    net = NeuralNetClassifier(
    module = model,
    criterion= torch.nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    module__embeddings=embeddings,
    module__vocabSize=vocab_size,
    optimizer__weight_decay = decay,
    optimizer__lr = lr,
    batch_size = 8,
    max_epochs=5,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,

    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score,average="macro",zero_division=0.0),
    'recall': make_scorer(recall_score,average="macro",zero_division=0.0),
    'f1_score': make_scorer(f1_score,average="macro",zero_division=0.0)
    }
    
    #run model
    cv_results = cross_validate(net, X_data, Y_data, cv=kf, scoring=scoring)

    #print results
    print("Model Accuracy: {0:.2%} Precision: {1:.4f} Recall: {2:.4f} F1 Score: {3:.4f}".format(np.mean(cv_results['test_accuracy']),np.mean(cv_results['test_f1_score']),np.mean(cv_results['test_precision']),np.mean(cv_results['test_recall']))) 


