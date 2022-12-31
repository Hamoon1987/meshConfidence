# python3 worstJoint/worstJoint.py --checkpoint=/SPINH/data/model_checkpoint.pt --dataset=3dpw --log_freq=20


import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from tqdm import tqdm
import sys
sys.path.insert(0, '/SPINH')
import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
import itertools
from utils.geometry import perspective_projection
from pytorchopenpose.src.body import Body
from utils.imutils import transform


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='3dpw', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=20 , type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, help='Batch size for testing')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')

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

def run_evaluation(model, dataset_name, dataset,
                   batch_size=2, log_freq=20):
    
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14

    eval_list = np.zeros(len(dataset))
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # step = 0
        # batch = next(itertools.islice(data_loader, step, None))
        # if step == 10:
        #     break
        # Get 3D GT from labels (accurate)
        gt_label_3d = batch['joint_position'].to(device)
        gt_label_3d = gt_label_3d.reshape(-1, 24, 3)
        joint_mapper_gt_label_3d = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]
        gt_label_3d = gt_label_3d[:, joint_mapper_gt_label_3d, :]
        gt_label_2d = get_2d_projection(batch, gt_label_3d)
        gt_label_2d = torch.tensor(gt_label_2d, dtype=torch.float).to(device)
        # Get GT from SMPL mesh (inaccurate) we need it just for the nose
        gender = batch['gender'].to(device)
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_joints_smpl = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints
        gt_joints_smpl_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints 
        gt_joints_smpl[gender==1, :, :] = gt_joints_smpl_female[gender==1, :, :].to(device).float()
    
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
        
        # Get 14 predicted joints from the mesh
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_keypoints_3d_ = pred_keypoints_3d[:, joint_mapper_h36m, :]
        # Replacing regressed estimates for hips, head and neck with the SMPL-regressed estimates 
        pred_keypoints_3d_[:, 2, :] = pred_joints[:, 9, :] #RH
        pred_keypoints_3d_[:, 3, :] = pred_joints[:, 12, :] #LH
        pred_keypoints_3d_[:, 13, :] = pred_joints[:, 15, :] #Head
        pred_keypoints_3d_[:, 12, :] = pred_joints[:, 40, :] #Neck

        # 2D projection of SPIN pred_keypoints and gt_keypoints
        focal_length = constants.FOCAL_LENGTH
        camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d_,
                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                            translation=camera_translation,
                                            focal_length=focal_length,
                                            camera_center=camera_center)
        gt_keypoints_2d = perspective_projection(gt_joints_smpl,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)
        

        
        # gt_keypoints_2d_n is the GT regressed from the mesh. Not accurate but has all the joints
        # gt_label_2d_n is the GT from the labels. Accurate but does not have all the joints
        # Use gt_label_2d_n to put the gt_keypoints_2d_n in the correct place. Based on the neck joint which is common in both.
        gt_keypoints_2d = gt_keypoints_2d - gt_keypoints_2d[:, 1:2, :] + gt_label_2d[:,12:13,:]
        
        # Replacing GT labels for head and neck with the SMPL-regressed GT from the mesh
        gt_label_2d[:, 12, :] = gt_keypoints_2d[:, 40, :]
        gt_label_2d[:, 13, :] = gt_keypoints_2d[:, 0, :]
        
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
                candidate = np.vstack([candidate, [constants.IMG_RES, constants.IMG_RES, 0, -1]])
                candidate_sorted = candidate[subset_sorted]
                candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float).to(device)
                error_s = torch.sqrt(((pred_keypoints_2d[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                subset_error.append(error_s)
            subset_index = subset_error.index(min(subset_error))        
            
            subset_sorted = subset[subset_index][map_op_smpl].astype(int)
            candidate = np.vstack([candidate, [constants.IMG_RES, constants.IMG_RES, 0, -1]])
            candidate_sorted = candidate[subset_sorted]
            op_confidence = torch.tensor(candidate_sorted[:,2:3], dtype=torch.float)
            candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
            candidate_sorted_list.append(candidate_sorted_t)
            op_confidence_list.append(op_confidence)
        candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)
        op_confidence_t = torch.stack(op_confidence_list, dim=0).to(device)

        sp_op = torch.sqrt(((pred_keypoints_2d - candidate_sorted_t) ** 2).sum(dim=-1)).to(device)
        sp_gt = torch.sqrt(((pred_keypoints_2d - gt_label_2d) ** 2).sum(dim=-1)).to(device)
        sp_op_max_ind = torch.argmax(sp_op, dim=1)
        sp_gt_max_ind = torch.argmax(sp_gt, dim=1)
        a = False
        counter = 0
        while(a==False):
            counter += 1
            a = True
            for i, v in enumerate(sp_op_max_ind):
                if (op_confidence_t[i, v] == 0):
                    a = False
                    sp_op[i, v] = -1
                    sp_op_max_ind = torch.argmax(sp_op, dim=1)
            if counter == 10:
                break
        eval = torch.eq(sp_op_max_ind, sp_gt_max_ind)
        eval = eval.cpu().numpy()
        eval_list[step * batch_size:step * batch_size + curr_batch_size] = eval

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            print('Eval: ' + str(100 * eval_list[:step * batch_size].mean()))
            print()
            original_img = images_[0]
            gt_label_2d = gt_label_2d[0]
            pred_keypoints_2d = pred_keypoints_2d[0]
            candidate_sorted_t = candidate_sorted_t[0]
            sp_op_max_ind = sp_op_max_ind[0]
            sp_gt_max_ind = sp_gt_max_ind[0]
            for i in [sp_op_max_ind, sp_gt_max_ind]:
                cv2.circle(original_img, (int(gt_label_2d[i][0]), int(gt_label_2d[i][1])), 3, color = (0, 255, 0), thickness=-1) 
                cv2.circle(original_img, (int(pred_keypoints_2d[i][0]), int(pred_keypoints_2d[i][1])), 3, color = (255, 0, 0), thickness=-1) 
                cv2.circle(original_img, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 3, color = (0, 0, 255), thickness=-1) 
            cv2.imwrite(f"worstJoint/test_{step}.jpg", original_img)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('Eval: ' + str(100 * eval_list.mean()))



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
    run_evaluation(model, args.dataset, dataset,
                    batch_size=args.batch_size,
                    log_freq=args.log_freq)