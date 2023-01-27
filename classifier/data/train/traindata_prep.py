import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd

dataset = "h36m_p1"
occluded = True
load_path = "sp_op/" + dataset + "/" + dataset + "_train_"
if occluded:
    load_path = "sp_op/" + dataset + "/" + dataset + "_occ_train_"
sp_op = np.load(load_path + 'sp_op.npy')
sp_gt = np.load(load_path + 'sp_gt.npy')
# labels for sp_gt that are less than 0.1 get label 1 rest 0
treshold = 0.1
label_wj = np.argmax(sp_gt, axis=1)
sp_gt_max = np.amax(sp_gt, axis=1)
label_m = np.zeros(sp_gt.shape[0])
label_m[sp_gt_max <= treshold] = 1
df = pd.DataFrame(sp_op, columns = ['sp_op_0','sp_op_1','sp_op_2','sp_op_3','sp_op_4','sp_op_5','sp_op_6','sp_op_7','sp_op_8','sp_op_9','sp_op_10', 'sp_op_11','sp_op_12','sp_op_13'])
df["label_m"] = label_m
df["label_wj"] = label_wj
df['label_c'] = df["label_wj"]
df.loc[df['label_m'] == 1, 'label_c'] = 14

# shuffle the DataFrame rows and split into test and training
if dataset=="3dpw":
    df = df.sample(frac=1)
    df_val = df.sample(frac=0.2)
    df_train = df.drop(df_val.index)
else:
    df = df.sample(frac=1)
    df_test = df.sample(frac = 0.5)
    df_training = df.drop(df_test.index)
    df_val = df_training.sample(frac = 0.2)
    df_train = df_training.drop(df_val.index)
# Save the data
if occluded:
    df_train.to_csv(f'classifier/data/train/occ_{dataset}_train.csv', index=False)
    df_val.to_csv(f'classifier/data/val/occ_{dataset}_val.csv', index=False)
    if dataset=="h36m_p1":
        df_test.to_csv(f'classifier/data/test/occ_{dataset}_test.csv', index=False)
    
else:
    df_train.to_csv(f'classifier/data/train/{dataset}_train.csv', index=False)
    df_val.to_csv(f'classifier/data/val/{dataset}_val.csv', index=False)
    if dataset=="h36m_p1":
        df_test.to_csv(f'classifier/data/test/{dataset}_test.csv', index=False)
    