import numpy as np
import torch

occ_joint = True
if occ_joint:
    spin_pred_smpl = torch.tensor(np.load(f'worstJoint/occ_spin_pred_smpl.npy'))
    spin_pred_reg = torch.tensor(np.load(f'worstJoint/occ_spin_pred_reg.npy'))
    gt_label = torch.tensor(np.load(f'worstJoint/occ_gt_label.npy'))
    gt_mesh_smpl = torch.tensor(np.load(f'worstJoint/occ_gt_mesh_smpl.npy'))
    op = torch.tensor(np.load(f'worstJoint/occ_op.npy'))
    op_conf = torch.tensor(np.load(f'worstJoint/occ_op_conf.npy'))
else:
    spin_pred_smpl = torch.tensor(np.load(f'worstJoint/spin_pred_smpl.npy'))
    spin_pred_reg = torch.tensor(np.load(f'worstJoint/spin_pred_reg.npy'))
    gt_label = torch.tensor(np.load(f'worstJoint/gt_label.npy'))
    gt_mesh_smpl = torch.tensor(np.load(f'worstJoint/gt_mesh_smpl.npy'))
    op = torch.tensor(np.load(f'worstJoint/op.npy'))
    op_conf = torch.tensor(np.load(f'worstJoint/op_conf.npy'))

sp_op = torch.sqrt(((spin_pred_smpl - op) ** 2).sum(dim=-1))
sp_gt = torch.sqrt(((spin_pred_smpl - gt_label) ** 2).sum(dim=-1))


# ##### Model 1 GT 1
# sp_op_max_ind = torch.argmax(sp_op, dim=1)
# sp_gt_max_ind = torch.argmax(sp_gt, dim=1)
# eval = torch.eq(sp_op_max_ind, sp_gt_max_ind)
# eval = eval.cpu().numpy()
# print(100 * eval.mean())



###### Model 2 GT 1
sp_gt_max_ind = torch.argmax(sp_gt, dim=1)
_, sp_op_max_ind = torch.topk(sp_op, 3)
eval = torch.zeros(len(sp_gt_max_ind))
for i in range(len(sp_gt_max_ind)):
    if sp_gt_max_ind[i] in sp_op_max_ind[i]:
        eval[i] = 1
eval = eval.cpu().numpy()
print(100 * eval.mean())


######## Model 2 GT 2
# _, sp_op_max_ind = torch.topk(sp_op, 2)
# _, sp_gt_max_ind = torch.topk(sp_gt, 2)
# sp_op_max_ind = sp_op_max_ind.cpu().numpy()
# sp_gt_max_ind = sp_gt_max_ind.cpu().numpy()
# eval = np.zeros(len(sp_op_max_ind))
# for i in range(len(sp_op_max_ind)):
#     eval[i] = len(set(sp_op_max_ind[i]).intersection(set(sp_gt_max_ind[i])))
# print(50 * eval.mean())

###### Model 1 GT 2
# sp_op_max_ind = torch.argmax(sp_op, dim=1)
# _, sp_gt_max_ind = torch.topk(sp_gt, 2)
# eval = torch.zeros(len(sp_gt_max_ind))
# for i in range(len(sp_gt_max_ind)):
#     if sp_op_max_ind[i] in sp_gt_max_ind[i]:
#         eval[i] = 1
# eval = eval.cpu().numpy()
# print(100 * eval.mean())