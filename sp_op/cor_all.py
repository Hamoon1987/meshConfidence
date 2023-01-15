import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

dataset_index = 0
occluded = True
dataset_name = ["3dpw", "h36m-p1", "h36m-p2", "mpi-inf-3dhp", "3doh"]
dataset = dataset_name[dataset_index]
if occluded:
    path = "sp_op/" + dataset + "/" + dataset + "_occ_"
else:
    path = "sp_op/" + dataset + "/" + dataset + "_"
print(path)

sp_op_all = np.load(path + 'sp_op.npy')
sp_gt_all = np.load(path + 'sp_gt.npy')
op_conf_all = np.load(path + 'conf.npy')

sp_gt_all_mean = [a.mean() for a in sp_gt_all]
sp_op_all_mean = [a.mean() for a in sp_op_all]
my_rho_all = round(np.corrcoef(sp_gt_all_mean, sp_op_all_mean)[0,1], 2)
print("Correlation Coefficent low = ", my_rho_all)
# my_rho_l = round(np.corrcoef(sp_op_l, sp_gt_l)[0,1], 2)
# print("Correlation Coefficent low = ", my_rho_l)
# fig, ax = plt.subplots(figsize = (9, 9))
# ax.scatter(sp_op_l, sp_gt_l, s=1, c='r')
# # ax.scatter(sp_op[j], sp_gt[j], s=15, c='k')
# plt.xlim([0, 0.6])
# plt.ylim([0, 0.6])
# plt.xlabel('ED',fontsize=38)
# plt.ylabel('SE',fontsize=38)
# # We change the fontsize of minor ticks label 
# ax.tick_params(axis='both', which='major', labelsize=26)
# plt.xlim(left=0)
# plt.ylim(bottom=0)



# print()
# print("Average coefficient l= ", (np.mean(rho_list_l)))
# plt.savefig(path + f'ED_SE_{joint_index}.jpg')