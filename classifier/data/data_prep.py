import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd
from classifier_config import args

# dataset_name = ["3dpw", "h36m-p1", "h36m-p2", "3doh"]
dataset = "3dpw"
mode = "train"
occluded = False
split = False
if occluded:
    path = "sp_op/" + dataset + "/" + dataset + "_occ_" + mode + "_"
else:
    path = "sp_op/" + dataset + "/" + dataset + "_" + mode + "_"
print(path)

sp_op = np.load(path + 'sp_op.npy')
sp_gt = np.load(path + 'sp_gt.npy')

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


# if not occluded:
#     if split:
#         # shuffle the DataFrame rows and split into test and training 
#         df_train = df.sample(frac = args.tt_split_ratio)
#         df_test = df.drop(df_train.index)
#         # Save the data
#         df_train.to_csv(f'classifier/{dataset}_train.csv', index=False)
#         df_test.to_csv(f'classifier/{dataset}_test.csv', index=False)
#     elif not split :
#         df.to_csv(f'classifier/{dataset}_{mode}.csv', index=False)
# else:
#     if split:
#         # shuffle the DataFrame rows and split into test and training 
#         df_train = df.sample(frac = args.tt_split_ratio)
#         df_test = df.drop(df_train.index)
#         # Save the data
#         df_train.to_csv(f'classifier/occ_{dataset}_train.csv', index=False)
#         df_test.to_csv(f'classifier/occ_{dataset}_test.csv', index=False)
#     elif not split :
#         df.to_csv(f'classifier/occ_{dataset}_{mode}.csv', index=False)

# # Combine the train data
# df_3dpw = pd.read_csv("classifier/3dpw_train.csv")
# df_3dpw_occ = pd.read_csv("classifier/occ_3dpw_train.csv")
# df_h36m = pd.read_csv("classifier/h36m-p1_train.csv")
# df_h36m_occ = pd.read_csv("classifier/occ_h36m-p1_train.csv")
# df_all_train = pd.concat([df_3dpw, df_3dpw_occ, df_h36m, df_h36m_occ])
# df_all_train = df_all_train.sample(frac = 1)
# df_all_train.to_csv('classifier/all_train.csv', index=False)

# # Test
# df_all = pd.read_csv(f"classifier/all_train.csv")
# # print(df_all.shape)
# # print(df_all.sample(5))
# train_t = torch.tensor(df_all.values)
# train_mean = torch.mean(train_t, dim=0)
# train_var = torch.var(train_t, dim=0)
# train_norm = (train_t-train_mean)/torch.sqrt(train_var)
# print(train_mean)
# print(train_var)
# print(train_t[0,:])
# print(train_norm[0,:])
# from torchvision.transforms import Normalize
# train_norm_new = Normalize(mean= train_mean, std=train_var)
# a = train_norm_new(train_t)
# print(a[0,:])