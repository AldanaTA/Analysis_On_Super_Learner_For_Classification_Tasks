
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from skorch import NeuralNetClassifier
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,classification_report
import os
import pandas as pd
import gc
import numpy as np
from gensim.models import KeyedVectors
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

def getMetrics(targets,preds):

    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='micro',zero_division=0.0)
    acc = accuracy_score(targets, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

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
    if samp_size > 1:
        samp_size = 1
    if samp_size <=0.1:
      samp_size = .1

    if int(len(df) * samp_size) <= 5000:
        df = df.sample(n=int(len(df) * samp_size))
    else:
        df = df.sample(n=5000)

    df.reset_index(inplace=True,drop=True)
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

def run_experiments(filename,sample_size,min_df,max_df,lr,momentum,weight_decay,b_size):
    current_dir = os.getcwd() + '//embeds//'
    embeddings = KeyedVectors.load_word2vec_format(current_dir+'GoogleNews-vectors-negative300.bin.gz',binary=True)
    filename = "data_set_1.csv"
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
    optimizer = torch.optim.SGD,
    module__embeddings=embeddings,
    module__vocabSize=vocab_size,
    optimizer__weight_decay = .01,
    optimizer__lr = .001,
    optimizer__momentum = .001,
    batch_size = 8,
    max_epochs=5,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,

    )

    net.set_params(train_split=False, verbose=0)
    parameters = {
        'optimizer__weight_decay': weight_decay,
        'optimizer__lr': lr,
        'optimizer__momentum' : momentum,
        "batch_size": b_size
        
    }

    params = tuning_script.grid_search(net,parameters,X_data,Y_data)
    return params
    
def run_model(filename,device,sample_size,min_df,max_df,lr,decay,momentum):
    f_loss = []
    acc = []
    precision = []
    f1 = []
    recall = []
    #Getting Embeddings
    current_dir = os.getcwd() + '//embeds//'
    embeddings = KeyedVectors.load_word2vec_format(current_dir+'GoogleNews-vectors-negative300.bin.gz',binary=True)

    #Reading Data
    filename = "data_set_1.csv"
    X_data,Y_data,embeddings = prep_data(filename,sample_size,embeddings,min_df,max_df)
    embeddings = get_embeddings(embeddings)
    vocab_size = len(embeddings)
    embeddings = get_embeddings()

    #Setting embeddings
    model = RNNClassifier(embeddings=embeddings,vocabSize=vocab_size,device=device)
    kfold = KFold(shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X_data,Y_data)):
        print(f"FOLD {fold+1}")

        train_x = X_data[train_ids]
        train_y = Y_data.iloc[train_ids].astype(int).apply(encode_Train_Lables).to_numpy()

        val_x = X_data[test_ids]
        val_y = Y_data.iloc[test_ids].apply(encode_Test_Lables).to_numpy()
        
        train_y = np.stack(train_y)
      
        train_x = train_x.astype(np.int32)
        
        val_x = val_x.astype(np.int32)

        
        
        train_dataset = TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,pin_memory=True,num_workers=4)

        val_dataset = TensorDataset(torch.from_numpy(val_x),torch.from_numpy(val_y))
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True,pin_memory=True,num_workers=4)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=decay,momentum=momentum)

        #Training Loop
        for epoch in range(5):
            
            for text,lables in train_dataloader:
                text = text.to(device)
                lables = lables.to(device)

                optimizer.zero_grad()
        
                yhat = model(text)
                loss = criterion(yhat,lables)
                loss.backward()
                optimizer.step()
        #Results
        with torch.no_grad():
            totalLoss = 0
            preds = []
            targets = []
            for text,lables in val_dataloader:
       
                text = text.to(device)
                lables = lables.to(device)
        
                yhat = model(text)

                loss = criterion(yhat,lables)

                yhat = yhat.cpu()
                lables = lables.cpu()
                
                preds.extend(torch.argmax(yhat,dim=1).numpy())
                targets.extend(lables.numpy())
                totalLoss+=loss.item()
            results = getMetrics(targets,preds)
            acc.append(results['accuracy'])
            precision.append(results['precision'])
            f1.append(results['f1'])
            recall.append(results['recall'])
            f_loss.append(totalLoss/len(val_dataloader))
            print("Avg Total Loss: {0:.2f}".format(totalLoss/len(val_dataloader)))
            print(classification_report(targets,preds,zero_division=0.0))
    acc = np.array(acc)
    precision = np.array(precision)
    f1 = np.array(f1)
    recall = np.array(recall)
    f_loss = np.array(f_loss)

    print("Model Loss: {0:.4f} Accuracy {1:.2%} Precision: {2:.4f} Recall: {3:.4f} F1 Score: {4:.4f}".format(np.mean(f_loss),np.mean(acc),np.mean(precision),np.mean(recall),np.mean(f1),))

'''

filename = "data_set_1.csv"

sample_size = 5000

min_df = .001
max_df = .97

lr = [.01,.05]
momentum = [.01,.05]
weight_decay = [.001,.005]


run_experiments(filename,sample_size,min_df,max_df,lr,momentum,weight_decay)
'''

