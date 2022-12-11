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

def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

# Create dataloader for the dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
dataset = BaseDataset(None, "3dpw", is_train=False)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
""" Step """
step = 279
# 5400
# 279
batch = next(itertools.islice(data_loader, step, None))
print(batch["imgname"])
images = batch['img'].to(device)
batch_size = images.shape[0]
model = hmr(config.SMPL_MEAN_PARAMS)
checkpoint = torch.load("data/model_checkpoint.pt", map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
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
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
joint_mapper_h36m = constants.H36M_TO_J14

# SPIN Estimate
with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(images)
    pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices
# Get 14 predicted joints from the mesh
J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
pred_keypoints_3d_ = pred_keypoints_3d[:, joint_mapper_h36m, :]
pred_keypoints_3d = pred_keypoints_3d_ - pred_pelvis 

# Ground Truth from mesh
gt_pose = batch['pose'].to(device)
gt_betas = batch['betas'].to(device)
gender = batch['gender'].to(device)
gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
gt_keypoints_3d_ = gt_keypoints_3d[:, joint_mapper_h36m, :]
gt_keypoints_3d = gt_keypoints_3d_ - gt_pelvis

# 2D projection of points
focal_length = constants.FOCAL_LENGTH
camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
pred_keypoints_2d = perspective_projection(pred_keypoints_3d_,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)
gt_keypoints_2d = perspective_projection(gt_keypoints_3d_,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)

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

# Projecting GT and SPIN mesh
# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False).to(device)
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
back = np.zeros((constants.IMG_RES, constants.IMG_RES, 3))


gt_vertices = gt_vertices[0].cpu().numpy()
pred_vertices = pred_vertices[0].cpu().numpy()
pred_keypoints_2d = pred_keypoints_2d[0].cpu().numpy()
gt_keypoints_2d = gt_keypoints_2d[0].cpu().numpy()
camera_translation = camera_translation[0].cpu().numpy()
op = candidate_sorted_t[0]

original_img = denormalize(images)[0]
original_img = images_[0]
cv2.imwrite(f'sp_op/original_img.png', original_img)
img_mesh1 = renderer(gt_vertices, camera_translation, back, (255, 255, 255, 1))
img_mesh2 = renderer(pred_vertices, camera_translation, img_mesh1, (0.8, 0.3, 0.8, 1))
cv2.circle(img_mesh2, (int(op[6][0]), int(op[6][1])), 3, color = (0, 255, 0), thickness=-1) #OpenPose
cv2.imwrite("examples/testmesh.jpg", 255 * img_mesh2[:, : ,::-1])



