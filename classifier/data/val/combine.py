import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd

# Combine the val data
df_3dpw = pd.read_csv("classifier/data/val/3dpw_val.csv")
df_3dpw_occ = pd.read_csv("classifier/data/val/occ_3dpw_val.csv")
df_h36m = pd.read_csv("classifier/data/val/h36m_p1_val.csv")
df_h36m_occ = pd.read_csv("classifier/data/val/occ_h36m_p1_val.csv")
df_all_val = pd.concat([df_3dpw, df_3dpw_occ, df_h36m, df_h36m_occ])
df_all_val = df_all_val.sample(frac = 1)
df_all_val.to_csv('classifier/data/val/all_val.csv', index=False)