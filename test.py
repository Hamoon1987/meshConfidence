import numpy as np
# import mat73
# mat = mat73.loadmat('/SPINH/data/MPI_INF_3DHP/mpi_inf_3dhp_test_set/TS1/annot_data.mat')
# print(mat["annot3"])



a = np.load("/SPINH/data/dataset_extras/mpi_inf_3dhp_valid.npz")
print(a['S'][0].shape)
# print(a[10])
# import numpy as np
# import matplotlib.pyplot as plt
# sp_op = np.load('sp_op.npy') 
# sp_gt = np.load('mpjpe.npy')
# print(sp_op[5400])
# print(sp_gt[5400])
# # indexes of so_op that are less than 0.05
# sp_op_index = [i for i,v in enumerate(sp_op) if v < 0.05]
# # indexes of sp_gt that are more than 0.4
# sp_gt_index = [i for i,v in enumerate(sp_gt) if v > 0.4]
# # common indexes
# common_list = set(sp_op_index).intersection(sp_gt_index)
# print(common_list)
# print(np.mean(sp_gt)*1000)
# my_rho = np.corrcoef(sp_op, sp_gt)
# print(my_rho)

# fig, ax = plt.subplots(figsize = (9, 9))
# ax.scatter(sp_op, sp_gt, s=8)
# plt.xlabel('ED',fontsize=18)
# plt.ylabel('SE',fontsize=18)
# plt.xlim(left=0)
# plt.ylim(bottom=0)


# def line_fitting_errors(x, data):
#     m = x[0]
#     c = x[1]
#     y_predicted = m * data[:,0] + c
#     e = y_predicted - data[:,1]
#     return e

# theta_guess = [1, 1] # Need an intial guess
# data = np.vstack([sp_op, sp_gt]).T
# n = data.shape[0]

# # Non-linear least square fitting
# from scipy.optimize import least_squares
# retval = least_squares(line_fitting_errors, theta_guess, loss='linear', args=[data])
# print(f'Reasons for stopping: {retval["message"]}')
# print(f'Success: {retval["success"]}')
# theta = retval['x']
# print(f'data = {n}')
# print(f'theta = {theta}')


# # Create sequence of 100 numbers from 0 to 100 
# xseq = np.linspace(0, 1, num=4)

# # Plot regression line
# a=theta[1]
# b=theta[0]
# ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
# #add fitted regression equation to plot
# ax.text(0.8, 0.35, 'y = ' + '{:.2f}'.format(a) + ' + {:.2f}'.format(b) + 'x', size=18)
# ax.axhline(y=0.2, color='k', linestyle='--')
# ax.axvline(x=0.25, color='k', linestyle='--')
# ax.text(0.1, 0.3, "a", size=20, color='k')
# ax.text(0.35, 0.3, "b", size=20, color='k')
# ax.text(0.1, 0.1, "c", size=20, color='k')
# ax.text(0.35, 0.1, "d", size=20, color='k')
# plt.savefig('RightWrist.jpg')


# import math
# import torch
# import argparse
# import cv2
# from models import hmr, SMPL
# import config
# from datasets import BaseDataset
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import constants
# import itertools
# from tqdm import tqdm
# import pandas as pd

# df = pd.DataFrame(columns=['Name'])
# Name = []
# dataset_name = "3dpw"
# dataset = BaseDataset(None, dataset_name, is_train=False)
# batch_size = 1
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
#     Name.append(batch["imgname"][0])
# df['Name'] = Name
# df.to_csv('3dpw.csv', index=True)
# print(df.head())