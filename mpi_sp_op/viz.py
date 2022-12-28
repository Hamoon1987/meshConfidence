import numpy as np
import matplotlib.pyplot as plt

rho_list = []
rho_list_l = []
occluded = False
for joint_index in range (6, 7):# choose the joint index
    if occluded:
        sp_op = np.load(f'sp_op/occ_sp_op_{joint_index}.npy') 
        sp_gt = np.load(f'sp_op/occ_mpjpe_{joint_index}.npy')
        op_conf = np.load(f'sp_op/occ_conf_{joint_index}.npy')
    else:
        sp_op = np.load(f'mpi_sp_op/sp_op_{joint_index}.npy') 
        sp_gt = np.load(f'mpi_sp_op/mpjpe_{joint_index}.npy')
        op_conf = np.load(f'mpi_sp_op/conf_{joint_index}.npy')
        op_mpjpe = np.load(f'mpi_sp_op/op_mpjpe_{joint_index}.npy')
        mpjpe_2d = np.load(f'mpi_sp_op/mpjpe_2d_{joint_index}.npy')


    # limit results to high confidence OpenPose outputs
    op_conf_index = [i for i,v in enumerate(op_conf) if v > 0.9]
    sp_op_index = [i for i,v in enumerate(sp_op) if v < 0.1]
    sp_gt_index = [i for i,v in enumerate(sp_gt) if v > 0.3]
    common_list = set(sp_op_index).intersection(sp_gt_index)
    print(common_list)
    print(sp_op[110])
    print(sp_gt[110])
    print("joint index = ", joint_index)
    sp_op_l = sp_op[op_conf_index]
    sp_gt_l = sp_gt[op_conf_index]

    # print("SPIN error = ", np.mean(sp_gt)*1000)
    my_rho = np.corrcoef(sp_op, sp_gt)[0,1]
    my_rho_l = np.corrcoef(sp_op_l, sp_gt_l)[0,1]
    my_rho_op = np.corrcoef(op_conf, op_mpjpe)[0,1]
    print("Correlation Coefficent = ", my_rho)
    # print("Correlation Coefficent low = ", my_rho_l)
    print("Correlation Coefficent op = ", my_rho_op)
    fig, ax = plt.subplots(figsize = (9, 9))
    ax.scatter(sp_op, sp_gt, s=1, c='b')
    
    ax.scatter(sp_op_l, sp_gt_l, s=1, c='r')
#     ax.scatter(op_conf_l, op_mpjpe_l, s=1, c='b')
    plt.xlabel('ED',fontsize=18)
    plt.ylabel('SE',fontsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

#     plt.title(my_rho , size=20, color='k')
    if occluded:
        plt.savefig(f'mpi_sp_op/Occ_ED_SE_{joint_index}.jpg')
    else:
        plt.savefig(f'mpi_sp_op/ED_SE_{joint_index}.jpg')
#     rho_list.append(my_rho)
#     rho_list_l.append(my_rho_l)
# print()
# print("Average coefficient = ", (np.mean(rho_list)))
# print("Average coefficient l= ", (np.mean(rho_list_l)))