import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd
import os


for dataset in ["3dpw", "h36m_p1"]:
    for occluded in [False, True]:
        load_path = "sp_op/" + dataset + "/" + dataset + "_train_"
        if occluded:
            load_path = "sp_op/" + dataset + "/" + dataset + "_occ_train_"
        sp_op = np.load(load_path + 'sp_op.npy')
        sp_gt = np.load(load_path + 'sp_gt.npy')
        conf = np.load(load_path + "conf.npy")

        op_conf_lh = conf[:,2]
        op_conf_rh = conf[:,3]
        op_conf_t = conf[:,12]
        # limit results to high confidence OpenPose outputs
        op_conf_index_lh = set([i for i,v in enumerate(op_conf_lh) if v < 0.1])
        op_conf_index_rh = set([i for i,v in enumerate(op_conf_rh) if v < 0.1])
        op_conf_index_t = set([i for i,v in enumerate(op_conf_t) if v < 0.1])
        op_conf_index = list(set().union(op_conf_index_lh, op_conf_index_rh, op_conf_index_t))
        sp_op_l = [v for i,v in enumerate(sp_op) if i not in op_conf_index]
        sp_gt_l = [v for i,v in enumerate(sp_gt) if i not in op_conf_index]
        sp_gt_new, sp_op_new = [], []
        limit = np.ones((14,1))
        for i in range(len(sp_op_l)):
            a = sum(sp_gt_l[i] > 1)
            if a == 0:
                sp_gt_new.append(sp_gt_l[i])
                sp_op_new.append(sp_op_l[i])
        print(sp_op.shape)
        sp_op = np.array(sp_op_new)
        sp_gt = np.array(sp_gt_new)
        print(sp_op.shape)
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
        df = df.sample(frac=1)
        df_val = df.sample(frac=0.2)
        df_train = df.drop(df_val.index)

        # Save the data
        if occluded:
            df_train.to_csv(f'classifier/data/train/occ_{dataset}_train.csv', index=False)
            df_val.to_csv(f'classifier/data/val/occ_{dataset}_val.csv', index=False)
        else:
            df_train.to_csv(f'classifier/data/train/{dataset}_train.csv', index=False)
            df_val.to_csv(f'classifier/data/val/{dataset}_val.csv', index=False)

# Combine the train data
df_3dpw = pd.read_csv("classifier/data/train/3dpw_train.csv")
df_3dpw_occ = pd.read_csv("classifier/data/train/occ_3dpw_train.csv")
df_h36m = pd.read_csv("classifier/data/train/h36m_p1_train.csv")
df_h36m_occ = pd.read_csv("classifier/data/train/occ_h36m_p1_train.csv")
df_all_train = pd.concat([df_3dpw, df_3dpw_occ, df_h36m, df_h36m_occ])
df_all_train = df_all_train.sample(frac = 1)
df_all_train.to_csv('classifier/data/train/all_train.csv', index=False)
os.remove("classifier/data/train/3dpw_train.csv")
os.remove("classifier/data/train/occ_3dpw_train.csv")
os.remove("classifier/data/train/h36m_p1_train.csv")
os.remove("classifier/data/train/occ_h36m_p1_train.csv")

# Combine the val data
df_3dpw = pd.read_csv("classifier/data/val/3dpw_val.csv")
df_3dpw_occ = pd.read_csv("classifier/data/val/occ_3dpw_val.csv")
df_h36m = pd.read_csv("classifier/data/val/h36m_p1_val.csv")
df_h36m_occ = pd.read_csv("classifier/data/val/occ_h36m_p1_val.csv")
df_all_val = pd.concat([df_3dpw, df_3dpw_occ, df_h36m, df_h36m_occ])
df_all_val = df_all_val.sample(frac = 1)
df_all_val.to_csv('classifier/data/val/all_val.csv', index=False)
os.remove("classifier/data/val/3dpw_val.csv")
os.remove("classifier/data/val/occ_3dpw_val.csv")
os.remove("classifier/data/val/h36m_p1_val.csv")
os.remove("classifier/data/val/occ_h36m_p1_val.csv")

# Print the norm
df_all = pd.read_csv(f"classifier/data/train/all_train.csv")
train_t = torch.tensor(df_all.values)
train_mean = torch.mean(train_t, dim=0)
train_var = torch.var(train_t, dim=0)
print("train_mean",train_mean[:14])
print("train_var",train_var[:14])