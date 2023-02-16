import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

dataset_index = 1
occluded = False
dataset_name = ["3dpw", "h36m_p1", "h36m-p2", "mpi-inf-3dhp", "3doh"]
dataset = dataset_name[dataset_index]
if occluded:
    path = "sp_op/" + dataset + "/" + dataset + "_occ_test_"
else:
    path = "sp_op/" + dataset + "/" + dataset + "_train_"
print(path)

sp_op_all = np.load(path + 'sp_op.npy')
sp_gt_all = np.load(path + 'sp_gt.npy')
op_conf_all = np.load(path + 'conf.npy')
rho_list = []
rho_list_l = []

for joint_index in range (14):# choose the joint index
    sp_op = sp_op_all[:,joint_index]
    sp_gt = sp_gt_all[:,joint_index]
    op_conf_j = op_conf_all[:,joint_index]
    op_conf_lh = op_conf_all[:,2]
    op_conf_rh = op_conf_all[:,3]
    op_conf_t = op_conf_all[:,12]
    # limit results to high confidence OpenPose outputs
    op_conf_index_j = set([i for i,v in enumerate(op_conf_j) if v < 0.1])
    op_conf_index_lh = set([i for i,v in enumerate(op_conf_lh) if v < 0.1])
    op_conf_index_rh = set([i for i,v in enumerate(op_conf_rh) if v < 0.1])
    op_conf_index_t = set([i for i,v in enumerate(op_conf_t) if v < 0.1])
    op_conf_index = list(set().union(op_conf_index_j,op_conf_index_lh, op_conf_index_rh, op_conf_index_t))
    print("joint index = ", joint_index)
    # sp_op_l = sp_op[op_conf_index]
    # sp_gt_l = sp_gt[op_conf_index]
    sp_op_l = [v for i,v in enumerate(sp_op) if i not in op_conf_index]
    sp_gt_l = [v for i,v in enumerate(sp_gt) if i not in op_conf_index]

    # sp_op_index = [i for i,v in enumerate(sp_op) if (v < 0.1)]
    # sp_gt_index = [i for i,v in enumerate(sp_gt) if v > 0.2]
    # common_list = set(sp_op_index).intersection(sp_gt_index)
    # common_list = list(common_list)
    # common_list = [i for i in common_list if op_conf_j[i] > 0.1]
    # print(np.sort(common_list)[:50])
    # j=1143
    # print(sp_gt[j])
    # print(sp_op[j])
    # print(op_conf_j[j])
    # print(op_conf_lh[j])
    # print(op_conf_rh[j])
    # print(op_conf_t[j])

    # print("SPIN error = ", np.mean(sp_gt)*1000)
    # my_rho = round(np.corrcoef(sp_op, sp_gt)[0,1], 2)
    my_rho_l = round(np.corrcoef(sp_op_l, sp_gt_l)[0,1], 2)
    # my_rho_op = np.corrcoef(op_conf, op_mpjpe)[0,1]
    # print("Correlation Coefficent = ", my_rho)
    print("Correlation Coefficent low = ", my_rho_l)
    # print("Mean MPJPE low = ", np.mean(sp_gt_l))
    # print("Correlation Coefficent op = ", my_rho_op)
    fig, ax = plt.subplots(figsize = (9, 9))
    # ax.scatter(sp_op, sp_gt, s=2, c='b')
    ax.scatter(sp_op_l, sp_gt_l, s=1, c='r')
    # ax.scatter(sp_op[j], sp_gt[j], s=15, c='k')
    # ax.scatter(op_conf_l, op_mpjpe_l, s=1, c='b')
    # plt.xlim([0, 0.6])
    # plt.ylim([0, 0.6])
    plt.xlabel('ED',fontsize=38)
    plt.ylabel('SE',fontsize=38)
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    if joint_index == 6:
        ax.axhline(y=0.1, color='k', linestyle='--')
        ax.axvline(x=0.1, color='k', linestyle='--')
        ax.text(0.05, 0.15, "a", size=35, color='k', fontweight = 'bold')
        ax.text(0.15, 0.15, "b", size=35, color='k', fontweight = 'bold')
        ax.text(0.05, 0.05, "c", size=35, color='k', fontweight = 'bold')
        ax.text(0.15, 0.05, "d", size=35, color='k', fontweight = 'bold')
    
    ax.set_title(f'$r_{{{joint_index}}}$={my_rho_l}' , y=1.0, pad=-45, size=45)
    fig.set_tight_layout(tight=True)
    plt.savefig(path + f'ED_SE_{joint_index}.jpg')

    # rho_list.append(my_rho)
    rho_list_l.append(my_rho_l)

print()
# print("Average coefficient = ", (np.mean(rho_list)))
print("Average coefficient l= ", (np.mean(rho_list_l)))
joint_names = ['Right Ankle',
                'Right Knee',
                'Right Hip',
                'Left Hip',
                'Left Knee',
                'Left Ankle',
                'Right Wrist',
                'Right Elbow',
                'Right Shoulder',
                'Left Shoulder',
                'Left Elbow',
                'Left Wrist',
                'Neck',
                'Top of Head'
                ]
w, h = 7, 2
f, axarr = plt.subplots(h, w)
f.set_size_inches((w*3, h*3))

for jid in range(14):
    image = img.imread(path + f'ED_SE_{jid}.jpg')
    axarr[jid // w, jid % w].axis('off')
    axarr[jid // w, jid % w].set_title(
    f'{joint_names[jid]}'
    )
    axarr[jid // w, jid % w].imshow(image)

f.set_tight_layout(tight=True)
plt.savefig(path + f'ED_SE_All.jpg')