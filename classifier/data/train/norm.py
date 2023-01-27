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

train_mean = [0.1641, 0.1455, 0.0864, 0.0867, 0.1424, 0.1554, 0.3645, 0.2034, 0.1010, 0.1051, 0.1781, 0.3030, 0.0639, 0.6305]
train_std = [6.5141e-02, 5.4406e-02, 4.7398e-03, 4.5912e-03, 6.2194e-02, 5.9077e-02,3.5126e-01, 1.7716e-01, 1.4936e-02, 1.0990e-02, 1.4384e-01, 2.8130e-01, 1.8599e-03, 7.2290e-01]