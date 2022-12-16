import numpy as np
import matplotlib.pyplot as plt

rho_list = []
rho_list_l = []
occluded = False
for joint_index in range (14):# choose the joint index
    if occluded:
        sp_op = np.load(f'sp_op/occ_sp_op_{joint_index}.npy') 
        sp_gt = np.load(f'sp_op/occ_mpjpe_{joint_index}.npy')
        op_conf = np.load(f'sp_op/occ_conf_{joint_index}.npy')
    else:
        sp_op = np.load(f'sp_op/sp_op_{joint_index}.npy') 
        sp_gt = np.load(f'sp_op/mpjpe_{joint_index}.npy')
        op_conf = np.load(f'sp_op/conf_{joint_index}.npy')

    # limit results to high confidence OpenPose outputs
    op_conf_index = [i for i,v in enumerate(op_conf) if v < 0.5]
    print("joint index = ", joint_index)
    sp_op_l = sp_op[op_conf_index]
    sp_gt_l = sp_gt[op_conf_index]

    # print("SPIN error = ", np.mean(sp_gt)*1000)
    my_rho = np.corrcoef(sp_op, sp_gt)[0,1]
    my_rho_l = np.corrcoef(sp_op_l, sp_gt_l)[0,1]
    print("Correlation Coefficent = ", my_rho)
    print("Correlation Coefficent low = ", my_rho_l)

    fig, ax = plt.subplots(figsize = (9, 9))
    ax.scatter(sp_op, sp_gt, s=1, c='b')
    ax.scatter(sp_op_l, sp_gt_l, s=1, c='r')
    plt.xlabel('ED',fontsize=18)
    plt.ylabel('SE',fontsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)


    def line_fitting_errors(x, data):
        m = x[0]
        c = x[1]
        y_predicted = m * data[:,0] + c
        e = y_predicted - data[:,1]
        return e

    theta_guess = [1, 1] # Need an intial guess
    data = np.vstack([sp_op, sp_gt]).T
    n = data.shape[0]

    # Non-linear least square fitting
    from scipy.optimize import least_squares
    retval = least_squares(line_fitting_errors, theta_guess, loss='linear', args=[data])
    # print(f'Reasons for stopping: {retval["message"]}')
    # print(f'Success: {retval["success"]}')
    theta = retval['x']
    # print(f'data = {n}')
    # print(f'theta = {theta}')

    # Create sequence of 100 numbers from 0 to 100 
    xseq = np.linspace(0, 1, num=4)

    # Plot regression line
    a=theta[1]
    b=theta[0]
    ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
    #add fitted regression equation to plot
    ax.text(0.8, 0.35, 'y = ' + '{:.2f}'.format(a) + ' + {:.2f}'.format(b) + 'x', size=18)
    # ax.axhline(y=0.2, color='k', linestyle='--')
    # ax.axvline(x=0.25, color='k', linestyle='--')
    # ax.text(0.1, 0.3, "a", size=20, color='k')
    # ax.text(0.35, 0.3, "b", size=20, color='k')
    # ax.text(0.1, 0.1, "c", size=20, color='k')
    # ax.text(0.35, 0.1, "d", size=20, color='k')
    plt.title(my_rho , size=20, color='k')
    if occluded:
        plt.savefig(f'sp_op/Occ_ED_SE_{joint_index}.jpg')
    else:
        plt.savefig(f'sp_op/ED_SE_{joint_index}.jpg')
    rho_list.append(my_rho)
    rho_list_l.append(my_rho_l)
print()
print("Average coefficient = ", (np.mean(rho_list)))
print("Average coefficient l= ", (np.mean(rho_list_l)))