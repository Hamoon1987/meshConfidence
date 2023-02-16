"""
Provide a cropped and centered image and the results show the confidence on the mesh


```
Example with cropped and centered image
```
python3 demo/demo_confidence.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```
"""
import sys
sys.path.insert(0, '/SPINH')
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
from utils.renderer_m import Renderer_m
import config
import constants
from utils.geometry import perspective_projection
from pytorchopenpose.src.body import Body
import torch.nn as nn
from classifier import classifier_model
from classifier import classifier_wj_model


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
# parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def normalize(data):
        data_n = torch.sub(data, (constants.IMG_RES/2))
        data_n = torch.div(data_n, (constants.IMG_RES/2))
        return data_n

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

if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        smpl_pred_joints = pred_output.joints
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    focal_length = constants.FOCAL_LENGTH
    camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])

    # Get the 2D joint positions predicted by SPIN model
    smpl_pred_keypoints_2d = perspective_projection(smpl_pred_joints,
                                        rotation=torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1),
                                        translation=camera_translation,
                                        focal_length=focal_length,
                                        camera_center=camera_center)
    smpl_joint_map_op = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 40, 0]
    smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d[0, smpl_joint_map_op, :]
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()

    op_img = 255 * img[:,:,::-1]
    

    # Get the OpenPose joint predictions
        # OpenPose keypoints
    body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
    candidate, subset = body_estimation(op_img)
    # Map openpose to smpl 14 joints
    map_op_smpl = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0] 
      
    subset_sorted = subset[0][map_op_smpl].astype(int)
    candidate = np.vstack([candidate, [constants.IMG_RES, constants.IMG_RES, 0, -1]])
    candidate_sorted = candidate[subset_sorted]
    candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float).to(device)


    # Normalize the coordinates
    candidate_sorted_t_n = normalize(candidate_sorted_t)
    smpl_pred_keypoints_2d_op_n =normalize(smpl_pred_keypoints_2d_op)
    sp_op = torch.sqrt(((smpl_pred_keypoints_2d_op_n - candidate_sorted_t_n) ** 2).sum(dim=-1)).cpu().numpy()
    sp_op = torch.tensor(sp_op, dtype=torch.float)

    # Standardize
    mean = torch.tensor(constants.sp_op_NORM_MEAN, dtype=torch.float)
    std = torch.tensor(constants.sp_op_NORM_STD, dtype=torch.float)
    sp_op = sp_op.float()
    sp_op = (sp_op - mean)/torch.sqrt(std)


    # Load the classifier
    classifier = classifier_model()
    classifier.eval()
    classifier_wj = classifier_wj_model()
    classifier_wj.eval()
    renderer_m = Renderer_m(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    softmax = nn.Softmax(dim=0)
    with torch.no_grad():
        output = classifier(sp_op)
        output_wj = classifier_wj(sp_op)
        output_wj = softmax(output_wj)
        output = output[0].cpu().numpy()
        if output > 0.5:
            img_confidence_pred = renderer_m(pred_vertices, camera_translation.copy(), img, sp_op, norm=False)
            img_confidence_pred_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0,255,128,1))
        else:
            img_confidence_pred = renderer_m(pred_vertices, camera_translation.copy(), img, output_wj, norm=True)
            img_confidence_pred_mesh = renderer(pred_vertices, camera_translation.copy(), img, (0.8, 0.3, 0.3, 1))




    # Save reconstructions
    cv2.imwrite("demo/img_orig.png",  255 * img[:,:,::-1])
    cv2.imwrite("demo/img_WJC.png",  255 * img_confidence_pred[:,:,::-1])
    cv2.imwrite("demo/img_MC.png",  255 * img_confidence_pred_mesh[:,:,::-1])


    # for i in range(14):
    #     cv2.circle(op_img, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 3, color = (0,0,255), thickness=-1)
    #     cv2.circle(op_img, (int(smpl_pred_keypoints_2d_op[i][0]), int(smpl_pred_keypoints_2d_op[i][1])), 3, color = (255,0,0), thickness=-1)
    # cv2.imwrite("test.png", op_img)


    # # Render parametric shape
    # img_shape = renderer(pred_vertices, camera_translation, img, (0.3, 0.3, 0.8, 1))
    
    # # Render side views
    # aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    # center = pred_vertices.mean(axis=0)
    # rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    
    # # Render non-parametric shape
    # img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img), (0.3, 0.3, 0.8, 1))

    # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

    # # Save reconstructions
    # cv2.imwrite(outfile + '_confidence.png', 255 * img_shape[:,:,::-1])
    # cv2.imwrite(outfile + '_confidence_side.png', 255 * img_shape_side[:,:,::-1])
