# python3 op_eval.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw
# Evaluation for OpenPose model on 3DPW dataset. Uses 2D labels and projection to get GT.
# This code uses camera parameters to project the GT 3d joint positions
import time
import math
import torch
import argparse
import cv2
import config
from datasets import BaseDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import constants
from tqdm import tqdm
import matplotlib.pylab as plt
import os
from utils.imutils import crop
import itertools
from utils.geometry import perspective_projection
from utils.imutils import transform
from pytorchopenpose.src.body import Body
def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :, :, ::-1]
    return images

def get_gt_keypoints_2d(batch):

    # Prepare the required parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    camera_intrinsics = batch['camera_intrinsics'].to(device)
    camera_extrinsics = batch['camera_extrinsics'].to(device)
    joint_position = batch['joint_position'].to(device)
    joint_position = joint_position.reshape(-1, 24, 3)
    batch_size = joint_position.shape[0]
    # Preparing the regressor to map 24 3DPW keypoints on to 14 joints
    joint_mapper = [8, 5, 2, 1, 4, 7, 21, 19, 17,16, 18, 20, 12, 15]
    # Get 14 ground truth joints
    joint_position = joint_position[:, joint_mapper, :]

    # Project 3D keypoints to 2D keypoints
    # Homogenious real world coordinates X, P is the projection matrix
    P = torch.matmul(camera_intrinsics, camera_extrinsics).to(device)
    temp = torch.ones((batch_size, 14, 1)).double().to(device)
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
    new_p = np.ones((batch_size,14,2))
    for i in range(batch_size):
        for j in range(p.shape[1]):
            temp = transform(p[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
            new_p[i,j,:] = temp
    return new_p


def run_dataset(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # Load the dataloader
    dataset_name = args.dataset
    dataset = BaseDataset(None, dataset_name, is_train=False)
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    log_freq = args.log_freq

    # OpenPose 2d joint prediction
    body_estimation_model = Body('pytorchopenpose/model/body_pose_model.pth')
    mpjpe = np.zeros(len(dataset))
    for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        # Get occluded images
        gt_keypoints_2d = get_gt_keypoints_2d(batch)
        error = get_error(batch, gt_keypoints_2d, body_estimation_model)
        mpjpe[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] = error
        # Print intermediate results during evaluation
        if batch_idx % log_freq == log_freq - 1:
            print('MPJPE: ' + str(1000 * mpjpe[:batch_idx * batch_size].mean()))


    # Print final results during evaluation
    print('*** Final Results ***')
    print('mpjpe_occluded: ' + str(1000 * mpjpe.mean()))
    print()
    return 1000 * mpjpe.mean()

     


def get_error(batch, gt_keypoints_2d, body_estimation_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gt_keypoints_2d = torch.tensor(gt_keypoints_2d, device=device, dtype=torch.float)
    # Normalize between -1 and 1
    gt_keypoints_2d = torch.sub(gt_keypoints_2d, (constants.IMG_RES/2))
    gt_keypoints_2d = torch.div(gt_keypoints_2d, (constants.IMG_RES/2))
    # Relative position
    left_heap = gt_keypoints_2d[:,3:4,:].clone()
    left_heap = left_heap.expand(-1,14,-1)
    gt_keypoints_2d = gt_keypoints_2d - left_heap
    images = denormalize(batch['img'].to(device))
    curr_batch_size = images.shape[0]
    candidate_sorted_list = []
    for i in range(curr_batch_size):
        candidate, subset = body_estimation_model(images[i])
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
            error_s = torch.sqrt(((gt_keypoints_2d[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            subset_error.append(error_s)
        subset_index = subset_error.index(min(subset_error))        
        subset_sorted = subset[subset_index][map_op_smpl].astype(int)
        candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
        candidate_sorted = candidate[subset_sorted]
        candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
        candidate_sorted_list.append(candidate_sorted_t)
    candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)
    # Normalize between -1 and 1
    candidate_sorted_t = torch.sub(candidate_sorted_t, (constants.IMG_RES/2))
    candidate_sorted_t = torch.div(candidate_sorted_t, (constants.IMG_RES/2))
    # Relative position
    left_heap = candidate_sorted_t[:,3:4,:].clone()
    left_heap = left_heap.expand(-1,14,-1)
    candidate_sorted_t = candidate_sorted_t - left_heap

    error = torch.sqrt(((candidate_sorted_t - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    return error
    
if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None) # Path to network checkpoint
    parser.add_argument('--dataset', type=str, default='3dpw')  # Path of the input image
    parser.add_argument('--batch_size', default=1) # Batch size for testing
    parser.add_argument('--log_freq', default=50, type=int) # Frequency of printing intermediate results
    args = parser.parse_args()
    run_dataset(args)
    end = time.time()
    print("Time: ", end - start)