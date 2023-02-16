import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd



df_all = pd.read_csv(f"classifier/data/train/all_train.csv")
train_t = torch.tensor(df_all.values)
train_mean = torch.mean(train_t, dim=0)
train_var = torch.var(train_t, dim=0)
print(train_mean)
print(train_var)
# train_norm = (train_t-train_mean)/torch.sqrt(train_var)

train_mean = [0.1574, 0.1396, 0.0828, 0.0832, 0.1366, 0.1484, 0.3617, 0.2000, 0.0988,
        0.1021, 0.1727, 0.2965, 0.0620, 0.6291]
train_std = [5.8423e-02, 4.9447e-02, 2.2038e-03, 2.6574e-03, 5.7095e-02, 5.1747e-02,
        3.5120e-01, 1.7479e-01, 1.3567e-02, 9.0699e-03, 1.3944e-01, 2.7739e-01,
        1.1868e-03, 7.2544e-01]