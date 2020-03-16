import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, random, gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 256)
pd.set_option("display.max_rows", 256)

num_cols = ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"]
feature_num = 33
batch_size = 4
time_steps = 1
n_epocs = 300
lstm_hidden_dim = 32
target_dim = 1
NUM_MODELS = 5

class NeuralNet(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, target_dim):
        super(NeuralNet, self).__init__()
        self.input_dim = lstm_input_dim
        self.hidden_dim = lstm_hidden_dim
        self.lstm_1 = nn.LSTM(input_size=lstm_input_dim, 
                            hidden_size=lstm_hidden_dim,
                            dropout=0.2,
                            batch_first=True
                            )
        self.lstm_2 = nn.GRU(input_size=lstm_hidden_dim,
                            hidden_size=lstm_hidden_dim,
                            dropout=0.2,
                            batch_first=True
                            )
        self.linear = nn.Linear(lstm_hidden_dim, target_dim)

    def forward(self, X_input):
        h_lstm_1, _ = self.lstm_1(X_input)
        _, lstm_out = self.lstm_2(h_lstm_1)
        
        linear_out = self.linear(lstm_out[0].view(X_input.size(0), -1))
        return torch.sigmoid(linear_out)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def prep_feature_data(batch_idx, time_steps, X_data, feature_num, device):
    feats = torch.zeros((len(batch_idx), time_steps, feature_num), dtype=torch.float, device=device)
    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx + 1 - time_steps ,b_idx + 1)
        feats[b_i, :, :] = X_data[b_slc, :]

    return feats

def train_model(model, X_train, y_train, X_valid, y_valid):
    train_size = X_train.size(0)
    best_acc_score = 0
    for epoch in range(n_epocs):
        perm_idx = np.arange(0, train_size)
        for t_i in range(0, len(perm_idx), batch_size):
            batch_idx = perm_idx[t_i:(t_i + batch_size)]
            feats = prep_feature_data(batch_idx, time_steps, X_train, feature_num, device)
            y_target = y_train[batch_idx]
            model.zero_grad()
            train_scores = model(feats)
            loss = loss_function(train_scores, y_target.view(-1, 1))
            loss.backward()
            optimizer.step()

        print('EPOCH: ', str(epoch), ' loss :', loss.item())
        with torch.no_grad():
            feats_val = prep_feature_data(np.arange(time_steps, X_valid.size(0)), time_steps, X_valid, feature_num, device)
            val_scores = model(feats_val)
            tmp_scores = val_scores.view(-1).to('cpu').numpy()
            bi_scores = np.round(tmp_scores)
            acc_score = accuracy_score(y_valid[time_steps:], bi_scores)
            roc_score = roc_auc_score(y_valid[time_steps:], bi_scores)
            print('Val ACC Score :', acc_score, ' ROC AUC Score :', roc_score)

        if roc_score > best_acc_score:
            best_acc_score = roc_score
            torch.save(model.state_dict(),'./pytorch_v1.mdl')
            print('best score updated, Pytorch model was saved!!', )
        
    model.load_state_dict(torch.load('./pytorch_v1.mdl'))
    with torch.no_grad():
        feats_test = prep_feature_data(np.arange(0, X_test.size(0)), time_steps, X_test, feature_num, device)
        val_scores = model(feats_test)
        tmp_scores = val_scores.view(-1).to('cpu').numpy()
        bi_scores = np.round(tmp_scores)

        return bi_scores

if __name__ == "__main__":
    actual = pd.read_csv("../input/vortex/vortex_actual_04.csv")
    test = pd.read_csv("../input/vortex/vortex_test_04.csv")
    train = pd.read_csv("../input/vortex/vortex_train_04.csv")

    X_train, X_valid = train_test_split(train)
    y_train = X_train["target"].values
    X_train = X_train.loc[:, num_cols].values

    y_valid   = X_valid["target"].values
    X_valid = X_valid.loc[:, num_cols].values

    y_test = actual["target"].values
    X_test= test.loc[:, num_cols].values

    X_train = torch.tensor(X_train, dtype=torch.float, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float, device=device)

    X_valid = torch.tensor(X_valid, dtype=torch.float, device=device)

    X_test = torch.tensor(X_test, dtype=torch.float, device=device)

    all_test_preds = []
    for model_idx in range(NUM_MODELS):
        model = NeuralNet(feature_num, lstm_hidden_dim, target_dim).to(device)
        loss_function = nn.BCELoss()
        optimizer= optim.Adam(model.parameters(), lr=1e-4)
        all_test_preds.append(train_model(model, X_train, y_train, X_valid, y_valid))
        gc.collect()

    actual["pred"] = np.mean(all_test_preds, axis=0)
    actual.loc[actual["pred"]>=0.5, "pred"] = 1
    actual.loc[actual["pred"]<0.5, "pred"] = 0
    print(actual.loc[:, ["target", "pred"]])