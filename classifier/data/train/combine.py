import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd

# Combine the train data
df_3dpw = pd.read_csv("classifier/data/train/3dpw_train.csv")
df_3dpw_occ = pd.read_csv("classifier/data/train/occ_3dpw_train.csv")
df_h36m = pd.read_csv("classifier/data/train/h36m_p1_train.csv")
df_h36m_occ = pd.read_csv("classifier/data/train/occ_h36m_p1_train.csv")
df_all_train = pd.concat([df_3dpw, df_3dpw_occ, df_h36m, df_h36m_occ])
df_all_train = df_all_train.sample(frac = 1)
df_all_train.to_csv('classifier/data/train/all_train.csv', index=False)

# df = pd.read_csv("classifier/train/occ_h36m_p1_train.csv")
# print(df.shape)