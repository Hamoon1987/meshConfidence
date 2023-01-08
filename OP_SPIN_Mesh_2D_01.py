# You can run this code to choose an image in the 3DPW dataset and access the SPIN mesh, GT mesh, their 2D projection of keypoint and Openpose keypoints estimation. Combine these and generate images

# python3 OP_SPIN_Mesh_2D.py

from torch.utils.data import DataLoader
from datasets import BaseDataset
import itertools
import torch
import config
import constants
from models import hmr, SMPL
import numpy as np
from utils.geometry import perspective_projection
import cv2
from pytorchopenpose.src.body import Body
from utils.renderer import Renderer
from utils.renderer_m import Renderer_m
from utils.imutils import transform

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
    joint_position = joint_position.double().to(device)
    camera_intrinsics = batch['camera_intrinsics'].to(device)
    camera_extrinsics = batch['camera_extrinsics'].to(device)
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

def get_occluded_imgs(batch, joint_index):
    # Get the image batch find the ground truth joint location and occlude it. This file uses the ground truth 3d joint
    # positions and projects them.
    occ_size = 40
    occ_pixel = 0
    # joint_idx = args.occ_joint
    joint_idx = joint_index
    # print("Occluded joint", joint_idx)

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
    # Occlude the Images at the joint position
    images = batch['img'].to(device)
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
    
    return occ_images, new_p
# Create dataloader for the dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
dataset = BaseDataset(None, "3dpw", is_train=False)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
""" Step """
step = 0
# 931
# 228
# 234
# 5400
# 279
batch = next(itertools.islice(data_loader, step, None))
print(batch["imgname"])


occluded = False
occ_joint_index = 0
occ_imgs, new_p = get_occluded_imgs(batch, occ_joint_index)
if occluded:
    batch['img'] = occ_imgs
images = batch['img'].to(device)

batch_size = images.shape[0]
model = hmr(config.SMPL_MEAN_PARAMS)
checkpoint = torch.load("data/model_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(device)
# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=1,
                         create_transl=False).to(device)
smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                    create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR,
                    gender='male',
                    create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR,
                    gender='female',
                    create_transl=False).to(device)
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
joint_mapper_h36m = constants.H36M_TO_J14

# SPIN Estimate
with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(images)
    pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    # pred_vertices = pred_output.vertices
    pred_joints_new = pred_output.joints

joint_position = batch['joint_position'].to(device)
joint_position = joint_position.reshape(-1, 24, 3)
# Preparing the regressor to map 24 3DPW keypoints on to 14 joints
joint_mapper = [8, 5, 2, 1, 4, 7, 21, 19, 17,16, 18, 20, 12, 15]
# Get 14 ground truth joints
joint_position = joint_position[:, joint_mapper, :]
joint_position_2d = get_2d_projection(batch, joint_position)
joint_position_2d = torch.tensor(joint_position_2d, dtype=torch.float).to(device)
# 2D projection of points
focal_length = constants.FOCAL_LENGTH
camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2]).to(device)
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)

smpl_pred_keypoints_2d = perspective_projection(pred_joints_new,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)

map_pred_smpl = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 40, 15]
smpl_pred_keypoints_2d_smpl = smpl_pred_keypoints_2d[:, map_pred_smpl, :]



# Get GT from SMPL mesh (inaccurate) we need it just for the nose
gender = batch['gender'].to(device)
gt_pose = batch['pose'].to(device)
gt_betas = batch['betas'].to(device)
gt_joints_smpl = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints
gt_joints_smpl_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).joints 
gt_joints_smpl[gender==1, :, :] = gt_joints_smpl_female[gender==1, :, :].to(device).float()
gt_joints_smpl_2d = perspective_projection(gt_joints_smpl,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)
gt_joints_smpl_2d = gt_joints_smpl_2d - gt_joints_smpl_2d[:, 37:38, :] + joint_position_2d[:,12:13,:]
joint_position_2d[:,12,:] = gt_joints_smpl_2d[:, 40, :]
joint_position_2d[:,13,:] = gt_joints_smpl_2d[:, 15, :]

# OpenPose Estimate
body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
# De-normalizing the image
images_ = denormalize(images)

# candidate is (n, 4) teh 4 columns are the x, y, confidence, counter.
# subset (1, 20) if joint is not found -1 else counter. The last element is the number of found joints 
candidate_sorted_list = []
for i in range(batch_size):
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
        candidate_sorted_t = torch.tensor(candidate_sorted[:,:3], dtype=torch.float).to(device)
        error_s = torch.sqrt(((smpl_pred_keypoints_2d_smpl[i] - candidate_sorted_t[:,:2]) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        subset_error.append(error_s)
    subset_index = subset_error.index(min(subset_error))        
    
    subset_sorted = subset[subset_index][map_op_smpl].astype(int)
    candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
    candidate_sorted = candidate[subset_sorted]
    candidate_sorted_t = torch.tensor(candidate_sorted[:,:3], dtype=torch.float)
    candidate_sorted_list.append(candidate_sorted_t)
candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)




original_img = images_[0]
smpl_pred_keypoints_2d_smpl = smpl_pred_keypoints_2d_smpl[0]
op = candidate_sorted_t[0]
joint_position_2d = joint_position_2d[0]
gt_joints_smpl_2d = gt_joints_smpl_2d[0]
# for i in range(49):
#     cv2.circle(original_img, (int(smpl_pred_keypoints_2d_gt[i][0]), int(smpl_pred_keypoints_2d_gt[i][1])), 3, color = (255, 0, 0), thickness=-1) #gt_joint_position
for i in range(joint_position_2d.shape[0]):
    cv2.circle(original_img, (int(smpl_pred_keypoints_2d_smpl[i][0]), int(smpl_pred_keypoints_2d_smpl[i][1])), 3, color = (255, 0, 0), thickness=-1) #gt_joint_position
    cv2.circle(original_img, (int(op[i][0]), int(op[i][1])), 1, color = (0, 0, 255), thickness=-1) #OpenPose
    cv2.circle(original_img, (int(joint_position_2d[i][0]), int(joint_position_2d[i][1])), 1, color = (0, 255, 0), thickness=-1) #OpenPose
    # cv2.circle(original_img, (int(pred_keypoints_2d[i][0]), int(pred_keypoints_2d[i][1])), 1, color = (255, 255, 255), thickness=-1) #OpenPose


# if occluded:
#     cv2.imwrite(f'examples/occ_test.png', original_img)
# else:
#     cv2.imwrite(f'examples/test.png', original_img)

# cv2.imwrite("examples/testmesh.jpg", 255 * img_mesh1[:, : ,::-1])
cv2.imwrite("examples/testmesh.jpg", original_img)


