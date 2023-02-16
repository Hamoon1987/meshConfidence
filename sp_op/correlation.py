# This code gets the SPIN and OpenPose model estimations and calculates SP-OP and SP-GT and OP-Confidence. Afterwards saves the results for further investigation. You can choose to add occlusion and whether align the predictions before subtracting. 
# python3 sp_op/correlation.py --checkpoint=/SPINH/data/model_checkpoint.pt --dataset=h36m-p2 --log_freq=20


import sys
sys.path.insert(0, '/SPINH')
from datasets import BaseDataset
from torch.utils.data import DataLoader
import itertools
import cv2
import torch
import constants
from pytorchopenpose.src.body import Body
import numpy as np
import config
from models import hmr, SMPL
from utils.geometry import perspective_projection
import argparse
from tqdm import tqdm
from utils.imutils import transform
import random
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p2', choices=['h36m-p1', 'h36m-p2', '3dpw', '3doh', 'mpi-inf-3dhp'], help='Choose  dataset')
parser.add_argument('--log_freq', default=20 , type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')
parser.add_argument('--occ_size', type=int, default='40') # Size of occluding window
parser.add_argument('--pixel', type=int, default='0') # Occluding window - pixel values

def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

def normalize(data):
        data_n = torch.sub(data, (constants.IMG_RES/2))
        data_n = torch.div(data_n, (constants.IMG_RES/2))
        return data_n

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

def get_occluded_imgs(batch, args, new_p):
    # Get the image batch find the ground truth joint location and occlude it. This file uses the ground truth 3d joint
    # positions and projects them.
    occ_size = args.occ_size
    occ_pixel = args.pixel
    joint_idx = random.randint(0,13)
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
    
def run_evaluation(model, dataset_name, dataset,
                   batch_size=16, img_res=224, 
                   num_workers=1, shuffle=False, log_freq=20):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    # Transfer model to the GPU
    model.to(device)
    sp_gt = np.zeros((len(dataset), 14))
    sp_op = np.zeros((len(dataset), 14))
    op_conf = np.zeros((len(dataset), 14))
    occ_joint = False
    relative = True
    if occ_joint:
        path = "sp_op/" + dataset_name + "/" + dataset_name + "_occ_"
    else:
        path = "sp_op/" + dataset_name + "/" + dataset_name + "_"
    print(path)
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        # 2D GT from labels
        if dataset_name == "3dpw":
            gt_label_3d = batch['joint_position'].to(device)
            gt_label_3d = gt_label_3d.reshape(-1, 24, 3)
            joint_mapper_gt_label_3d = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15, 6]
            gt_label_3d = gt_label_3d[:, joint_mapper_gt_label_3d, :]
            gt_label_2d = get_2d_projection(batch, gt_label_3d)
            gt_label_2d = torch.tensor(gt_label_2d, dtype=torch.float).to(device)
            gt_spine_2d = gt_label_2d[:, [-1], :]
            gt_keypoints_2d = gt_label_2d[:,:-1,:]
        elif (dataset_name == "h36m-p2" or dataset_name == "h36m-p1"):
            S_2D = batch['S_2D']
            keywords_map=[0,1,2,3,4,5,6,7,8,9,10,11,12,17,16] # spine is 16
            S_2D = S_2D[:, keywords_map,:2] * 1000 # 1000 is for denormalizing (Initial size 1000)
            center = batch['center']
            scale = batch['scale']
            res = [constants.IMG_RES, constants.IMG_RES]
            gt_keypoints_2d = np.ones((curr_batch_size,S_2D.shape[1],2))
            for i in range(curr_batch_size):
                for j in range(S_2D.shape[1]):
                    temp = transform(S_2D[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
                    gt_keypoints_2d[i,j,:] = temp
            gt_keypoints_2d = torch.tensor(gt_keypoints_2d, dtype=torch.float).to(device)
            gt_spine_2d = gt_keypoints_2d[:, [-1], :].clone()
            gt_keypoints_2d = gt_keypoints_2d[:,:-1,:]
        elif dataset_name == "mpi-inf-3dhp":
            keypoints = batch["keypoints"]
            keypoints = (keypoints + 1) * constants.IMG_RES/2
            map = [25,26,27,28,29,30,31,32,33,34,35,36,37,42,41] # 41 is spine
            gt_keypoints_2d = keypoints[:,map,:2].to(device)
            gt_spine_2d = gt_keypoints_2d[:, [-1], :].clone()
            gt_keypoints_2d = gt_keypoints_2d[:,:-1,:]
        elif dataset_name == "3doh":
            keypoints = batch["keypoints"]
            keypoints = (keypoints + 1) * constants.IMG_RES/2
            map = [33, 30, 27, 26, 29, 32, 46, 44, 42, 41, 43, 45, 37, 40, 31] # 31 is spine
            gt_keypoints_2d = keypoints[:,map,:2].to(device)
            gt_spine_2d = gt_keypoints_2d[:, [-1], :].clone()
            gt_keypoints_2d = gt_keypoints_2d[:,:-1,:]
        # 2D predicted keypoint
        if occ_joint:
            images = get_occluded_imgs(batch, args, gt_keypoints_2d)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            smpl_pred_joints = pred_output.joints
        focal_length = constants.FOCAL_LENGTH
        camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        smpl_pred_keypoints_2d = perspective_projection(smpl_pred_joints,
                                            rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                            translation=camera_translation,
                                            focal_length=focal_length,
                                            camera_center=camera_center)
        pred_spine_2d = smpl_pred_keypoints_2d[:, [41],:].clone()
        if relative:
            smpl_pred_keypoints_2d = smpl_pred_keypoints_2d - pred_spine_2d + gt_spine_2d
        smpl_joint_map_op = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 40, 0] 
        smpl_joint_map_gt = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 37, 42]
        if (dataset_name == "h36m-p2" or dataset_name == "h36m-p1") :
            smpl_joint_map_op = [11, 10, 27, 28, 13, 14, 4, 3, 2, 5, 6, 7, 40, 0] 
            smpl_joint_map_gt = [11, 10, 27, 28, 13, 14, 4, 3, 2, 5, 6, 7, 37, 42]
        smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d[:, smpl_joint_map_op, :]
        smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d[:, smpl_joint_map_gt, :]

        # OpenPose keypoints
        body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
        image_ = denormalize(images)
        candidate_sorted_list = []
        op_confidence_list = []
        for i in range(curr_batch_size):
            candidate, subset = body_estimation(image_[i])
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
                error_s = torch.sqrt(((smpl_pred_keypoints_2d_op[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
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
        op_confidence_t = op_confidence_t.squeeze(2)


        op_spine = (((candidate_sorted_t[:, [2], :] + candidate_sorted_t[:, [3], :]) / 2) + 1.3*candidate_sorted_t[:, [12], :]) / 2.3
        if relative:
            candidate_sorted_t = candidate_sorted_t - op_spine + gt_spine_2d

        candidate_sorted_t_n = normalize(candidate_sorted_t)
        smpl_pred_keypoints_2d_op_n =normalize(smpl_pred_keypoints_2d_op)
        smpl_pred_keypoints_2d_gt_n = normalize(smpl_pred_keypoints_2d_gt)
        gt_keypoints_2d_n = normalize(gt_keypoints_2d)

        # Absolute error SPIN (MPJPE)
        error = torch.sqrt(((smpl_pred_keypoints_2d_gt_n - gt_keypoints_2d_n) ** 2).sum(dim=-1)).cpu().numpy()
        sp_gt[step * batch_size:step * batch_size + curr_batch_size] = error
        # SPIN - OpenPose (sp_op)
        error_ = torch.sqrt(((smpl_pred_keypoints_2d_op_n - candidate_sorted_t_n) ** 2).sum(dim=-1)).cpu().numpy()
        sp_op[step * batch_size:step * batch_size + curr_batch_size] = error_
        # OpenPose Confidence
        op_confidence_joint = op_confidence_t.cpu().numpy()
        op_conf[step * batch_size:step * batch_size + curr_batch_size] = op_confidence_joint


        # Visualize
        candidate_sorted_t = candidate_sorted_t[0]
        gt_keypoints_2d = gt_keypoints_2d[0]
        smpl_pred_keypoints_2d = smpl_pred_keypoints_2d[0]
        smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d_gt[0]
        smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d_op[0]
        image_test = image_[0]
        for i in range(14):
            # i=joint_index
            cv2.circle(image_test, (int(gt_keypoints_2d[i][0]), int(gt_keypoints_2d[i][1])), 3, color = (0, 255, 0), thickness=-1)
            cv2.circle(image_test, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 2, color = (0, 0, 255), thickness=-1)
            cv2.circle(image_test, (int(smpl_pred_keypoints_2d_gt[i][0]), int(smpl_pred_keypoints_2d_gt[i][1])), 2, color = (255, 0, 0), thickness=-1)
            cv2.circle(image_test, (int(smpl_pred_keypoints_2d_op[i][0]), int(smpl_pred_keypoints_2d_op[i][1])), 2, color = (255, 255, 255), thickness=-1)
        cv2.imwrite(f'sp_op/test.png', image_test)


    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    print('sp_gt: ' + str(1000 * sp_gt.mean()))
    print()
    print('sp_op: ' + str(1000 * sp_op.mean()))
    print()
    print('op_confidence: ' + str(op_conf.mean()))
    print()

    np.save(path +'test_sp_op.npy', sp_op) # save
    np.save(path +'test_sp_gt.npy', sp_gt) # save
    np.save(path +'test_conf.npy', op_conf) # save


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
    run_evaluation(model, args.dataset, dataset,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq)