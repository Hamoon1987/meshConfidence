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

def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
dataset = BaseDataset(None, "mpi-inf-3dhp", is_train=False)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
step = 0
#702
batch = next(itertools.islice(data_loader, step, None))




# 2D GT keypoints labels
keypoints = batch["keypoints"][0].to(device)
# keypoints = keypoints[keypoints[:,2]==1].to(device)
keypoints_ = (keypoints + 1) * constants.IMG_RES//2
keywords_map=[25,26,27,28,29,30,31,32,33,34,35,36,37,42]
# keywords_map=[0,1,2,3,4,5,6,7,8,9,10,11,12,15]
keypoints = keypoints_[keywords_map,:]

# 2D predicted keypoint
image = batch['img'].to(device)
print(batch['imgname'])
smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
model = hmr(config.SMPL_MEAN_PARAMS)
checkpoint="/SPINH/data/model_checkpoint.pt"
checkpoint = torch.load(checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(device)
with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(image)
    pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices
    smpl_pred_joints = pred_output.joints
# Get 14 predicted joints from the mesh
# J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float().to(device)
joint_mapper_h36m = constants.H36M_TO_J17
pred_keypoints_3d = torch.matmul(J_regressor, pred_vertices)
pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
pred_keypoints_3d_ = pred_keypoints_3d[:, joint_mapper_h36m, :]
# pred_keypoints_3d_ = pred_keypoints_3d_ - pred_pelvis

# Get 14 predicted joints from the mesh
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
test = torch.matmul(J_regressor_batch, pred_vertices)
print(test.shape)


# 2D projection of SPIN pred_keypoints and gt_keypoints
focal_length = constants.FOCAL_LENGTH
camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
smpl_pred_keypoints_2d = perspective_projection(smpl_pred_joints,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)
pred_keypoints_2d = perspective_projection(pred_keypoints_3d_,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)

test_2d = perspective_projection(test,
                                    rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                                    translation=camera_translation,
                                    focal_length=focal_length,
                                    camera_center=camera_center)
smpl_joint_map_op = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 40, 0]
smpl_joint_map_gt = [11, 10, 27, 28, 13, 14, 4, 3, 2, 5, 6, 7, 37, 42]
# smpl_pred_keypoints_2d = smpl_pred_keypoints_2d - smpl_pred_keypoints_2d[:, 39:40, :] + keypoints_[39:40, :2]
smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d[0, smpl_joint_map_op, :]
smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d[0, smpl_joint_map_gt, :]
# smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d[0]
pred_keypoints_2d = pred_keypoints_2d[0]
# smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d[0]
# smpl_pred_keypoints_2d = smpl_pred_keypoints_2d[0]


# # 3D labels

# pose_3d = batch["pose_3d"].to(device)
# pose_3d_map=[0,1,2,3,4,5,6,7,8,9,10,11,12,15]
# pose_3d = pose_3d[:, :, :-1]
# a=pose_3d[:, [13], :]
# print(a)
# # pose_3d = pose_3d + 

# pred_pose_2d = perspective_projection(pose_3d,
#                                     rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
#                                     translation=camera_translation,
#                                     focal_length=focal_length,
#                                     camera_center=camera_center)
# pred_pose_2d = pred_pose_2d + constants.IMG_RES//2 - pred_pose_2d[:, 13, :]
# OpenPose keypoints
image_ = denormalize(image)[0]
body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
candidate, subset = body_estimation(image_)
map_op_smpl = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
subset_sorted = subset[0][map_op_smpl].astype(int)
candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
candidate_sorted = candidate[subset_sorted]
candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
# pred_pose_2d = pred_pose_2d[0]
smpl_pred_keypoints_2d = smpl_pred_keypoints_2d[0]
# keypoints_ = keypoints_[0]
test_2d = test_2d[0]
for i in range(49):
    cv2.circle(image_, (int(smpl_pred_keypoints_2d[i][0]), int(smpl_pred_keypoints_2d[i][1])), 3, color = (255, 255, 255), thickness=-1)
for i in range(14):
#     i=0
#     cv2.circle(image_, (int(test_2d[i][0]), int(test_2d[i][1])), 1, color = (0, 0, 255), thickness=-1)
    cv2.circle(image_, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 1, color = (0, 0, 255), thickness=-1)
#     cv2.circle(image_, (int(smpl_pred_keypoints_2d_op[i][0]), int(smpl_pred_keypoints_2d_op[i][1])), 2, color = (255, 0, 0), thickness=-1)
    # cv2.circle(image_, (int(smpl_pred_keypoints_2d_gt[i][0]), int(smpl_pred_keypoints_2d_gt[i][1])), 3, color = (255, 255, 255), thickness=-1)

cv2.imwrite(f'sp_op/test.png', image_)