"""
Predict the 2d location of an specific joint with SPIN and OpenPose. Compare the difference and SPIN Error
Example usage:
```
python3 sp_op/sp_op1.py --checkpoint=/SPINH/data/model_checkpoint.pt --dataset=3dpw --log_freq=20
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
import sys
sys.path.insert(0, '/SPINH')
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
parser.add_argument('--occ_size', type=int, default='40')  # Size of occluding window
parser.add_argument('--pixel', type=int, default='0')  # Occluding window - pixel values
# parser.add_argument('--occ_joint', type=int, default='6')  # The joint you want to occlude

def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

def get_2d_projection(batch, joint_position):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    joint_position = joint_position.to(device)
    camera_intrinsics = batch['camera_intrinsics'].to(device)
    camera_extrinsics = batch['camera_extrinsics'].to(device)
    batch_size = camera_intrinsics.shape[0]
    P = torch.matmul(camera_intrinsics, camera_extrinsics).to(device)
    temp = torch.ones((batch_size, joint_position.shape[1], 1)).double().to(device)
    X = torch.cat((joint_position, temp), 2)
    X = X.permute(0, 2, 1)
    p = torch.matmul(P, X)
    p = torch.div(p[:,:,:], p[:,2:3,:])
    p = p[:, [0,1], :]
    # Projected 2d coordinates on image p with the shape of (batch_size, 14, 2)
    p = p.permute(0, 2, 1).cpu().numpy()
    # Process 2d keypoints to match the processed images in the dataset
    center = batch['center'].to(device)
    scale = batch['scale'].to(device)
    res = [constants.IMG_RES, constants.IMG_RES]
    new_p = np.ones((batch_size,joint_position.shape[1],2))
    for i in range(batch_size):
        for j in range(p.shape[1]):
            temp = transform(p[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
            new_p[i,j,:] = temp
    return new_p

def get_occluded_imgs(batch, args, new_p, joint_index):
    # Get the image batch find the ground truth joint location and occlude it. This file uses the ground truth 3d joint
    # positions and projects them.
    occ_size = args.occ_size
    occ_pixel = args.pixel
    joint_idx = joint_index
    # Prepare the required parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Occlude the Images at the joint position
    images = batch['img'].to(device)
    batch_size = images.shape[0]
    occ_images = images.clone()
    img_size = int(images[0].shape[-1])
    for i in range(batch_size):
        h_start = int(max(new_p[i, joint_idx, 1] - occ_size/2, 0))
        w_start = int(max(new_p[i, joint_idx, 0] - occ_size/2, 0))
        h_end = min(img_size, h_start + occ_size)
        w_end = min(img_size, w_start + occ_size)
        occ_images[i,0,h_start:h_end, w_start:w_end] = occ_pixel 
        occ_images[i,1,h_start:h_end, w_start:w_end] = occ_pixel
        occ_images[i,2,h_start:h_end, w_start:w_end] = occ_pixel
    return occ_images

def run_evaluation(model, joint_index, dataset_name, dataset, result_file,
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
    op_conf = np.zeros(len(dataset))
    # Choose if you want to occlude a joint before pose estimation
    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get 3D GT from labels (accurate) and project to 2D
        gt_label_3d = batch['joint_position'].to(device)
        gt_label_3d = gt_label_3d.reshape(-1, 24, 3)
        joint_mapper_gt_label_3d = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]
        gt_label_3d = gt_label_3d[:, joint_mapper_gt_label_3d, :]
        gt_label_2d = get_2d_projection(batch, gt_label_3d)
        gt_label_2d = torch.tensor(gt_label_2d, dtype=torch.float).to(device)

        occ_joint = False
        occ_joint_index = joint_index
        # Get the prediction
        if occ_joint:
            batch['img'] = get_occluded_imgs(batch, args, gt_label_2d, occ_joint_index)
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            smpl_pred_joints = pred_output.joints
        
        # 2D projection of SPIN pred_keypoints and gt_keypoints
        focal_length = constants.FOCAL_LENGTH
        camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        smpl_pred_keypoints_2d = perspective_projection(smpl_pred_joints,
                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                            translation=camera_translation,
                                            focal_length=focal_length,
                                            camera_center=camera_center)
        map_pred_smpl = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 40, 15]
        smpl_pred_keypoints_2d_smpl = smpl_pred_keypoints_2d[:, map_pred_smpl, :]

        # 2D projection of GT keypoints from the mesh
        gender = batch['gender'].to(device)
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_joints_smpl = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints
        gt_joints_smpl_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints 
        gt_joints_smpl[gender==1, :, :] = gt_joints_smpl_female[gender==1, :, :].to(device).float()
        gt_joints_smpl_2d = perspective_projection(gt_joints_smpl,
                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                            translation=camera_translation,
                                            focal_length=focal_length,
                                            camera_center=camera_center)
        gt_joints_smpl_2d = gt_joints_smpl_2d - gt_joints_smpl_2d[:, 1:2, :] + gt_label_2d[:,12:13,:]
        gt_label_2d[:,12,:] = gt_joints_smpl_2d[:, 40, :]
        gt_label_2d[:,13,:] = gt_joints_smpl_2d[:, 0, :]

        
        # OpenPose 2d joint prediction
        body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
        # De-normalizing the image
        images_ = denormalize(images)
        # candidate is (n, 4) teh 4 columns are the x, y, confidence, counter.
        # subset (1, 20) if joint is not found -1 else counter. The last element is the number of found joints 
        candidate_sorted_list = []
        op_confidence_list = []
        for i in range(curr_batch_size):
            candidate, subset = body_estimation(images_[i])
            if subset.shape[0] == 0:
                a = np.zeros((14,2))
                b = np.zeros((14,1))
                candidate_sorted_list.append(torch.tensor(a, dtype=torch.float))
                op_confidence_list.append(torch.tensor(b, dtype=torch.float))
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
                error_s = torch.sqrt(((smpl_pred_keypoints_2d_smpl[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                subset_error.append(error_s)
            subset_index = subset_error.index(min(subset_error))        
            
            subset_sorted = subset[subset_index][map_op_smpl].astype(int)
            candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
            candidate_sorted = candidate[subset_sorted]
            op_confidence = torch.tensor(candidate_sorted[:,2:3], dtype=torch.float)
            candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
            candidate_sorted_list.append(candidate_sorted_t)
            op_confidence_list.append(op_confidence)
        candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)
        op_confidence_t = torch.stack(op_confidence_list, dim=0).to(device)

        # Normalize between -1 and 1
        candidate_sorted_t_n = torch.sub(candidate_sorted_t, (constants.IMG_RES/2))
        candidate_sorted_t_n = torch.div(candidate_sorted_t_n, (constants.IMG_RES/2))
        smpl_pred_keypoints_2d_smpl_n = torch.sub(smpl_pred_keypoints_2d_smpl, (constants.IMG_RES/2))
        smpl_pred_keypoints_2d_smpl_n = torch.div(smpl_pred_keypoints_2d_smpl_n, (constants.IMG_RES/2))
        gt_label_2d_n = torch.sub(gt_label_2d, (constants.IMG_RES/2))
        gt_label_2d_n = torch.div(gt_label_2d_n, (constants.IMG_RES/2))

        # Absolute error SPIN (MPJPE)
        error = torch.sqrt(((smpl_pred_keypoints_2d_smpl_n[:, joint_index, :] - gt_label_2d_n[:, joint_index, :]) ** 2).sum(dim=-1)).cpu().numpy()
        mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

        # SPIN - OpenPose (sp_op)
        error_ = torch.sqrt(((smpl_pred_keypoints_2d_smpl_n[:, joint_index, :] - candidate_sorted_t_n[:, joint_index, :]) ** 2).sum(dim=-1)).cpu().numpy()
        sp_op[step * batch_size:step * batch_size + curr_batch_size] = error_

        # OpenPose Confidence
        op_confidence_joint = op_confidence_t[:, joint_index, 0].cpu().numpy()
        op_conf[step * batch_size:step * batch_size + curr_batch_size] = op_confidence_joint

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
            print('sp_op: ' + str(1000 * sp_op[:step * batch_size].mean()))
            print()


    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print()
    print('sp_op: ' + str(1000 * sp_op.mean()))
    print()
    print('op_confidence: ' + str(op_conf.mean()))
    print()

    if occ_joint:
        np.save(f'sp_op/occ_sp_op_{joint_index}.npy', sp_op) # save
        np.save(f'sp_op/occ_mpjpe_2d_{joint_index}.npy', mpjpe) # save
        np.save(f'sp_op/occ_conf_{joint_index}.npy', op_conf) # save
    else:
        np.save(f'sp_op/sp_op_{joint_index}.npy', sp_op) # save
        np.save(f'sp_op/mpjpe_2d_{joint_index}.npy', mpjpe) # save
        np.save(f'sp_op/conf_{joint_index}.npy', op_conf) # save

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
    print(len(dataset))
    # Run evaluation
    for i in range(13,14):
        print("joint_index = ", i)
        joint_index = i    
        run_evaluation(model, joint_index, args.dataset, dataset, args.result_file,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq)