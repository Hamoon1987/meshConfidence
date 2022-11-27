# python3 occlusion_mesh
# This code projects a predefined error array on the mesh and changes the colors accordingly
import torch
from models import SMPL
from utils.renderer_m import Renderer_m
import config
import constants
import numpy as np
import cv2
import matplotlib.pylab as plt

""" Joint index    joint_names = ['Right Ankle','Right Knee', 'Right Hip','Left Hip','Left Knee','Left Ankle','Right Wrist','Right Elbow',
                                                    'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck', 'Top of Head']"""

error_joint = [100.95, 106.4, 108.1, 107.25, 107.29, 101.32, 110.39, 112.64, 110.39, 111.9, 113, 111, 129.8, 125.18]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pred_rotmat = torch.eye(3)
pred_rotmat = pred_rotmat.unsqueeze(0).expand(24, 3, 3).unsqueeze(0).to(device)
pred_rotmat[0,0,1,1] = -1
pred_rotmat[0,0,2,2] = -1

pred_betas = torch.tensor([[ 0.5, 1, 0.5, 1, 0, 0, -0.3, 0.3, 0.2, -0.2]], device = device)
pred_camera = torch.tensor([[ 1, 0,  0.2]], device = device)
img_res = 960

# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
            batch_size=1,
            create_transl=False).to(device)

# Setup renderer for visualization
renderer = Renderer_m(focal_length=constants.FOCAL_LENGTH, img_res=img_res, faces=smpl.faces)


# Preprocess input image and generate predictions
with torch.no_grad():
    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices
# Calculate camera parameters for rendering
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(img_res * pred_camera[:,0] +1e-9)],dim=-1)
camera_translation = camera_translation[0].cpu().numpy()
pred_vertices = pred_vertices[0].cpu().numpy()

# Define the background
img = np.zeros((img_res, img_res, 3))

# Render parametric shape
img_mesh = renderer(pred_vertices, camera_translation, img, error_joint)
img_mesh = 255 * img_mesh[:,:,::-1]
# Save reconstructions
# cv2.imwrite('test.png', 255 * img_mesh[:,:,::-1])
plt.imshow(img_mesh)
# plt.imshow(img_mesh, alpha=1, cmap='jet', interpolation='none')
plt.colorbar()
plt.savefig('examples/mpjpe_mean.png')