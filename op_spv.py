"""
Predict the 2d location of an specific joint with SPIN and OpenPose. Compare the difference and SPIN Error
Example usage:
```
python3 op_spv.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
import itertools
from utils.geometry import perspective_projection
from pytorchopenpose.src.body import Body
from utils.imutils import transform
from utils.geometry import estimate_translation_np
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=20 , type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=1, shuffle=False, log_freq=20):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    sp_op = np.zeros(len(dataset))
    dp = np.zeros(len(dataset))

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # if step == 10:
        #     break
        """ Step """
        # step = 13
        # batch = next(itertools.islice(data_loader, step, None))
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                gt_keypoints_3d = batch['pose_3d'].to(device)
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d_ = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d_ - gt_pelvis


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d_ = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d_ - pred_pelvis 



            # 2D projection of SPIN pred_keypoints
            focal_length = constants.FOCAL_LENGTH
            camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
            camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
            pred_keypoints_2d = perspective_projection(pred_keypoints_3d_,
                                                rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                                translation=camera_translation,
                                                focal_length=focal_length,
                                                camera_center=camera_center)

            # Normalize between -1 and 1
            pred_keypoints_2d_n = torch.sub(pred_keypoints_2d, (constants.IMG_RES/2))
            pred_keypoints_2d_n = torch.div(pred_keypoints_2d_n, (constants.IMG_RES/2))

            # Relative position
            left_heap = pred_keypoints_2d_n[:,3:4,:].clone()
            left_heap = left_heap.expand(-1,14,-1)
            pred_keypoints_2d_n = pred_keypoints_2d_n - left_heap
            # OpenPose 2d joint prediction
            body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
            # De-normalizing the image
            images_ = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images_ = images_ + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            images_ = images_.permute(0, 2, 3, 1).cpu().numpy()
            images_ = 255 * images_[:, :,:,::-1]
            
            # candidate is (n, 4) teh 4 columns are the x, y, confidence, counter.
            # subset (1, 20) if joint is not found -1 else counter. The last element is the number of found joints 
            candidate_sorted_list = []
            for i in range(curr_batch_size):
                candidate, subset = body_estimation(images_[i])
                if subset.shape[0] == 0:
                    a = np.zeros((14,2))
                    candidate_sorted_list.append(torch.tensor(a, dtype=torch.float))
                    continue
                # Map openpose to smpl 14 joints
                map_op_smpl = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
                # Choose the right person in multiple people images for OpenPose
                subset_error = []
                for j in range(subset.shape[0]):
                    subset_sorted = subset[j][map_op_smpl].astype(int)
                    candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
                    candidate_sorted = candidate[subset_sorted]
                    candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float).to(device)
                    error_s = torch.sqrt(((pred_keypoints_2d[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                    subset_error.append(error_s)
                subset_index = subset_error.index(min(subset_error))        
                
                subset_sorted = subset[subset_index][map_op_smpl].astype(int)
                candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
                candidate_sorted = candidate[subset_sorted]
                candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
                candidate_sorted_list.append(candidate_sorted_t)
            candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)

            # Normalize between -1 and 1
            candidate_sorted_t_n = torch.sub(candidate_sorted_t, (constants.IMG_RES/2))
            candidate_sorted_t_n = torch.div(candidate_sorted_t_n, (constants.IMG_RES/2))

            # Relative position
            left_heap = candidate_sorted_t_n[:,3:4,:].clone()
            left_heap = left_heap.expand(-1,14,-1)
            candidate_sorted_t_n = candidate_sorted_t_n - left_heap

            # for i in range(subset_sorted.shape[0]): # OpenPose predicted keypoints (Blue)
            #     cv2.circle(images_[0], (int(candidate_sorted_t[0][i][0]), int(candidate_sorted_t[0][i][1])), 3, color = (255, 0, 0), thickness=1)

            # for i in range(pred_keypoints_2d.shape[1]): # SPIN predicted keypoints (Green)
            #     cv2.circle(images_[0], (int(pred_keypoints_2d[0][i][0]), int(pred_keypoints_2d[0][i][1])), 3, color = (0, 255, 0), thickness=1)

            # for i in range(new_p.shape[1]): # Label
            #     cv2.circle(images_[0], (int(new_p[0][i][0]), int(new_p[0][i][1])), 3, color = (255, 255, 255), thickness=1)

            # cv2.imwrite(f"op_sp_{step}.jpg", images_[0])

            # Absolute error SPIN (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d[:, 6, :] - gt_keypoints_3d[:, 6, :]) ** 2).sum(dim=-1)).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # SPIN - OpenPose (sp_op)
            error_ = torch.sqrt(((pred_keypoints_2d_n[:, 6, :] - candidate_sorted_t_n[:, 6, :]) ** 2).sum(dim=-1)).cpu().numpy()
            sp_op[step * batch_size:step * batch_size + curr_batch_size] = error_


            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print()
                print('sp_op: ' + str(1000 * sp_op[:step * batch_size].mean()))
                print()


    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print()
    print('sp_op: ' + str(1000 * sp_op.mean()))
    print()



    np.save('sp_op.npy', sp_op) # save
    np.save('mpjpe.npy', mpjpe) # save
    my_rho = np.corrcoef(sp_op, mpjpe)
    print(my_rho)

    # sp_op_norm = [float(i)/max(sp_op[:100]) for i in sp_op[:100]]
    # mpjpe_norm = [float(i)/max(mpjpe[:100]) for i in mpjpe[:100]]
    fig, ax = plt.subplots(figsize = (9, 9))
    ax.scatter(sp_op, mpjpe)
    plt.xlabel('SPIN-OP',fontsize=18)
    plt.ylabel('SPIN-GT',fontsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)



    

    def line_fitting_errors(x, data):
        m = x[0]
        c = x[1]
        y_predicted = m * data[:,0] + c
        e = y_predicted - data[:,1]
        return e

    theta_guess = [1, 1] # Need an intial guess
    data = np.vstack([sp_op, mpjpe]).T
    n = data.shape[0]


    # Non-linear least square fitting
    from scipy.optimize import least_squares
    retval = least_squares(line_fitting_errors, theta_guess, loss='arctan', args=[data])
    print(f'Reasons for stopping: {retval["message"]}')
    print(f'Success: {retval["success"]}')
    theta = retval['x']
    print(f'data = {n}')
    print(f'theta = {theta}')

    # # Fit linear regression via least squares with numpy.polyfit
    # # It returns an slope (b) and intercept (a)
    # # deg=1 means linear fit (i.e. polynomial of degree 1)
    # b, a = np.polyfit(sp_op[:1000], mpjpe[:1000], deg=1)

    # Create sequence of 100 numbers from 0 to 100 
    xseq = np.linspace(0, 1, num=4)

    # Plot regression line
    a=theta[1]
    b=theta[0]
    ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
    #add fitted regression equation to plot
    ax.text(0.25, 0.25, 'y = ' + '{:.2f}'.format(a) + ' + {:.2f}'.format(b) + 'x', size=14)
    plt.savefig('RightWrist.jpg')
if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)