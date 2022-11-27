
# python3 sp_op_mesh.py
# This code calculates the OpenPose confidence of an image and projects the confidence on the mesh

import torch
from torch.utils.data import DataLoader
from datasets import BaseDataset
import itertools
import numpy as np
import cv2
import os
from models import hmr, SMPL
import config
import constants
from utils.renderer import Renderer
from utils.renderer_m import Renderer_m
from torchvision.transforms import Normalize
from utils.imutils import crop
from pytorchopenpose.src.body import Body

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint_path = '/SPINH/data/model_checkpoint.pt'
# Load pretrained model
model = hmr(config.SMPL_MEAN_PARAMS).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(device)

def process_image(img_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False).to(device)

img_file = "/SPINH/sp_op/img01.jpg"
occ_img_file = "/SPINH/sp_op/img01.jpg"
img, norm_img = process_image(img_file, input_res=constants.IMG_RES)
occ_img, occ_norm_img = process_image(occ_img_file, input_res=constants.IMG_RES)
img = img.permute(1,2,0).cpu().numpy()
occ_img = occ_img.permute(1,2,0).cpu().numpy()


with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices
with torch.no_grad():
    occ_pred_rotmat, occ_pred_betas, occ_pred_camera = model(occ_norm_img.to(device))
    occ_pred_output = smpl(betas=occ_pred_betas, body_pose=occ_pred_rotmat[:,1:], global_orient=occ_pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    occ_pred_vertices = occ_pred_output.vertices

# Calculate camera parameters for rendering
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
camera_translation = camera_translation[0].cpu().numpy()
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float().to(device)
pred_keypoints_3d = torch.matmul(J_regressor, pred_vertices)
pred_pelvis = pred_keypoints_3d[0, 0,:].clone()
pred_vertices = pred_vertices[0].cpu().numpy()
pred_pelvis = pred_pelvis.cpu().numpy()


occ_camera_translation = torch.stack([occ_pred_camera[:,1], occ_pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * occ_pred_camera[:,0] +1e-9)],dim=-1)
occ_camera_translation = occ_camera_translation[0].cpu().numpy()
occ_pred_keypoints_3d = torch.matmul(J_regressor, occ_pred_vertices)
occ_pred_pelvis = occ_pred_keypoints_3d[0, 0,:].clone()
occ_pred_vertices = occ_pred_vertices[0].cpu().numpy()
occ_pred_pelvis = occ_pred_pelvis.cpu().numpy()
alignment = occ_pred_pelvis - pred_pelvis
occ_pred_vertices = occ_pred_vertices - alignment


# Setup renderer for visualization
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
renderer_m = Renderer_m(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

# OpenPose 2d joint prediction
body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
op_occ_img, _ = process_image(occ_img_file, input_res=constants.IMG_RES)
op_occ_img = op_occ_img.permute(1, 2, 0).cpu().numpy()
op_occ_img = 255 * op_occ_img[:,:,::-1]
candidate, subset = body_estimation(op_occ_img)
# Map openpose to smpl 14 joints
map_op_smpl = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
subset_sorted = subset[0][map_op_smpl].astype(int)
candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
candidate_sorted = candidate[subset_sorted]
confidence = candidate_sorted[:,2]

# Projecting the mesh and saving the images
back = np.zeros((constants.IMG_RES, constants.IMG_RES, 3))
for i in range(8):
    aroundy = cv2.Rodrigues(np.array([0, np.radians(45. * i), 0]))[0]
    center = pred_vertices.mean(axis=0)
    occ_center = occ_pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    occ_rot_vertices = np.dot((occ_pred_vertices - occ_center), aroundy) + occ_center
    img_mesh = renderer(rot_vertices, camera_translation, img, (255, 255, 255, 1))
    combine_mesh = renderer_m(occ_rot_vertices, occ_camera_translation, img_mesh, confidence)
    cv2.imwrite(f'sp_op/comb_img_mesh_{i}.png', 255 * combine_mesh[:, : ,::-1])
