import numpy as np
import torch
import sys
sys.path.insert(0, '/meshConfidence')


dataset_index = 0
occluded = True
dataset_name = ["3dpw", "h36m_p1", "h36m-p2", "mpi-inf-3dhp", "3doh"]
dataset = dataset_name[dataset_index]
if occluded:
    path = "sp_op/" + dataset + "/" + dataset + "_occ_test_"
else:
    path = "sp_op/" + dataset + "/" + dataset + "_test_"
print(path)
sp_op = np.load(path + 'sp_op.npy')
sp_gt = np.load(path + 'sp_gt.npy')
sp_op = torch.tensor(sp_op)
sp_gt = torch.tensor(sp_gt)
theta_list = np.load("line_fitting/theta.npy")
theta = torch.tensor(theta_list)
m = theta[:,0].unsqueeze(0)
c = theta[:,1].unsqueeze(0)




##### Model 1 GT 1
est_sp_op = m * sp_op + c
sp_op_max_ind = torch.argmax(est_sp_op, dim=1)
sp_gt_max_ind = torch.argmax(sp_gt, dim=1)
eval = torch.eq(sp_op_max_ind, sp_gt_max_ind)
eval = eval.cpu().numpy()
print(100 * eval.mean())


###### Model 3 GT 1
est_sp_op = m * sp_op + c
sp_gt_max_ind = torch.argmax(sp_gt, dim=1)
_, sp_op_max_ind = torch.topk(est_sp_op, 3)
eval = torch.zeros(len(sp_gt_max_ind))
for i in range(len(sp_gt_max_ind)):
    if sp_gt_max_ind[i] in sp_op_max_ind[i]:
        eval[i] = 1
eval = eval.cpu().numpy()
print(100 * eval.mean())

#### Mesh classifier
treshold = 0.1
sp_gt_max, _ = torch.max(sp_gt, dim=1)
label_m = torch.zeros(sp_gt.shape[0])
label_m[sp_gt_max <= treshold] = 1
est_sp_op = m * sp_op + c
sp_op_max, _ = torch.max(est_sp_op, dim=1)
pred_label_m = torch.zeros(est_sp_op.shape[0])
pred_label_m[sp_op_max <= treshold] = 1
eval = torch.eq(label_m, pred_label_m)
eval = eval.cpu().numpy()
print(100 * eval.mean())
