import numpy as np
import torch
import sys
sys.path.insert(0, '/SPINH')
import pandas as pd

dataset = "h36m_p1"
occluded = True
load_path = "sp_op/" + dataset + "/" + dataset + "_test_"
if occluded:
    load_path = "sp_op/" + dataset + "/" + dataset + "_occ_test_"
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


# Save the data
if occluded:
    df.to_csv(f'classifier/data/test/occ_{dataset}_test.csv', index=False)
else:
    df.to_csv(f'classifier/data/test/{dataset}_test.csv', index=False)

    