

import sys
sys.path.insert(0, '/SPINH')
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
from classifier import classifier_model
from classifier import classifier_wj_model
from sklearn.preprocessing import StandardScaler
from classifier.classifier_config import args
import torch.nn as nn



def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    # images = 255 * images[:, :,:,::-1]
    return images

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
dataset_index = 2
occluded = False
dataset_names = ["3dpw", "h36m_p1", "3doh"]
dataset_name = dataset_names[dataset_index]
dataset = BaseDataset(None, dataset_name, is_train=False)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
""" Step """
step = 0
batch = next(itertools.islice(data_loader, step, None))
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


# Load the classifier
classifier = classifier_model()
classifier.eval()
classifier_wj = classifier_wj_model()
classifier_wj.eval()

# Load sp-gt and sp-op
if occluded:
    path = "sp_op/" + dataset_name + "/" + dataset_name + "_occ_"
else:
    path = "sp_op/" + dataset_name + "/" + dataset_name + "_test_"
print(path)
sp_op = np.load(path + 'sp_op.npy')
sp_gt = np.load(path + 'sp_gt.npy')
sp_op = torch.tensor(sp_op)
sp_gt = torch.tensor(sp_gt)

# SPIN Estimate
with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(images)
    pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices

camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)

camera_translation = camera_translation[0].cpu().numpy()
pred_vertices = pred_vertices[0].cpu().numpy()
images = denormalize(images)
img = images[0]
sp_op = sp_op[step]
sp_gt = sp_gt[step]
# print("sp_op", sp_op)
# print("sp_gt", sp_gt)


renderer_m = Renderer_m(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
max = torch.max(sp_gt)
if max < 0.1:
    img_confidence_gt = renderer_m(pred_vertices, camera_translation.copy(), img, sp_gt, norm=False)
    img_confidence_gt_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0,255,128,1))
else:
    img_confidence_gt = renderer_m(pred_vertices, camera_translation.copy(), img, sp_gt, norm=True)
    img_confidence_gt_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0.8, 0.3, 0.3, 1))


# Standardize
mean = torch.tensor(constants.sp_op_NORM_MEAN, dtype=torch.float)
std = torch.tensor(constants.sp_op_NORM_STD, dtype=torch.float)
sp_op = sp_op.float()
sp_op = (sp_op - mean)/torch.sqrt(std)
input = sp_op.float()
softmax = nn.Softmax(dim=0)
with torch.no_grad():
    output = classifier(input)
    output_wj = classifier_wj(input)
    # output_wj = output_wj.cpu().numpy()
    # print("classifier_wj" , output_wj)
    output_wj = softmax(output_wj)
    # print("classifier_wj softamx" , output_wj)
    output = output[0].cpu().numpy()
    # print("classifier", output)
    if output > 0.5:
        img_confidence_pred = renderer_m(pred_vertices, camera_translation.copy(), img, sp_op, norm=False)
        img_confidence_pred_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0,255,128,1))
    else:
        img_confidence_pred = renderer_m(pred_vertices, camera_translation.copy(), img, output_wj, norm=True)
        img_confidence_pred_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0.8, 0.3, 0.3, 1))


# Save reconstructions
cv2.imwrite(f'qualitative/{dataset_name}_img_{step}_orig.png',  255 * img[:,:,::-1])
cv2.imwrite(f'qualitative/{dataset_name}_img_{step}_wj_confidence_pred.png',  255 * img_confidence_pred[:,:,::-1])
cv2.imwrite(f'qualitative/{dataset_name}_img_{step}_wj_confidence_gt.png',  255 * img_confidence_gt[:,:,::-1])
cv2.imwrite(f'qualitative/{dataset_name}_img_{step}_mesh_confidence_pred.png',  255 * img_confidence_pred_mesh[:,:,::-1])
cv2.imwrite(f'qualitative/{dataset_name}_img_{step}_mesh_confidence_gt.png',  255 * img_confidence_gt_mesh[:,:,::-1])
