import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.lines as mlines

sp_op_3dpw = np.load("sp_op/3dpw/3dpw_train_sp_op.npy")
sp_gt_3dpw = np.load("sp_op/3dpw/3dpw_train_sp_gt.npy")
conf_3dpw = np.load("sp_op/3dpw/3dpw_train_conf.npy")
sp_op_3dpw_occ = np.load("sp_op/3dpw/3dpw_occ_train_sp_op.npy")
sp_gt_3dpw_occ = np.load("sp_op/3dpw/3dpw_occ_train_sp_gt.npy")
conf_3dpw_occ = np.load("sp_op/3dpw/3dpw_occ_train_conf.npy")

sp_op_h36m_p1 = np.load("sp_op/h36m_p1/h36m_p1_train_sp_op.npy")
sp_gt_h36m_p1 = np.load("sp_op/h36m_p1/h36m_p1_train_sp_gt.npy")
conf_h36m_p1 = np.load("sp_op/h36m_p1/h36m_p1_train_conf.npy")
sp_op_h36m_p1_occ = np.load("sp_op/h36m_p1/h36m_p1_occ_train_sp_op.npy")
sp_gt_h36m_p1_occ = np.load("sp_op/h36m_p1/h36m_p1_occ_train_sp_gt.npy")
conf_h36m_p1_occ = np.load("sp_op/h36m_p1/h36m_p1_occ_train_conf.npy")


sp_op_all = np.concatenate((sp_op_3dpw, sp_op_3dpw_occ, sp_op_h36m_p1, sp_op_h36m_p1_occ), axis=0)
sp_gt_all = np.concatenate((sp_gt_3dpw, sp_gt_3dpw_occ, sp_gt_h36m_p1, sp_gt_h36m_p1_occ), axis=0)
conf_all = np.concatenate((conf_3dpw, conf_3dpw_occ, conf_h36m_p1, conf_h36m_p1_occ), axis=0)
# sp_op_all =sp_op_3dpw
# sp_gt_all = sp_gt_3dpw
# conf_all = conf_3dpw
theta_list =[]
for joint_index in range (14):# choose the joint index
    sp_op = sp_op_all[:,joint_index]
    sp_gt = sp_gt_all[:,joint_index]

    op_conf_j = conf_all[:,joint_index]
    op_conf_lh = conf_all[:,2]
    op_conf_rh = conf_all[:,3]
    op_conf_t = conf_all[:,12]
    # limit results to high confidence OpenPose outputs
    op_conf_index_j = set([i for i,v in enumerate(op_conf_j) if v < 0.1])
    op_conf_index_lh = set([i for i,v in enumerate(op_conf_lh) if v < 0.1])
    op_conf_index_rh = set([i for i,v in enumerate(op_conf_rh) if v < 0.1])
    op_conf_index_t = set([i for i,v in enumerate(op_conf_t) if v < 0.1])
    op_conf_index = list(set().union(op_conf_index_j,op_conf_index_lh, op_conf_index_rh, op_conf_index_t))
    print("joint index = ", joint_index)
    sp_op_l = [v for i,v in enumerate(sp_op) if i not in op_conf_index]
    sp_gt_l = [v for i,v in enumerate(sp_gt) if i not in op_conf_index]
    sp_gt_new, sp_op_new = [], []
    for i in range(len(sp_op_l)):
        if sp_gt_l[i] <= 1:
            sp_gt_new.append(sp_gt_l[i])
            sp_op_new.append(sp_op_l[i])
    sp_op = sp_op_new
    sp_gt = sp_gt_new
    fig, ax = plt.subplots(figsize = (9, 9))
    ax.scatter(sp_op, sp_gt, s=1, c='r')
    # plt.xlim([0, 0.6])
    # plt.ylim([0, 0.6])
    plt.xlabel('ED',fontsize=38)
    plt.ylabel('SE',fontsize=38)
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.xlim(left=0)
    plt.ylim(bottom=0)  
    fig.set_tight_layout(tight=True)

    theta = np.linalg.lstsq(np.vstack([sp_op, np.ones(len(sp_op))]).T, sp_gt, rcond=None)[0]
    def draw_line(theta):
        m = theta[0]
        c = theta[1]
        xmin, xmax = 0, 0.6
        ymin = m * xmin + c
        ymax = m * xmax + c
        l = mlines.Line2D([xmin, xmax], [ymin,ymax])
        ax.add_line(l)
    draw_line(theta)
    theta_list.append(theta)
    plt.savefig("line_fitting/" + f'ED_SE_{joint_index}.jpg')
print(theta_list)
np.save("line_fitting/theta.npy", theta_list) # save