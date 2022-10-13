# python3 occlusion_analysis03.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p2
# This runs through whole dataset and moves the occlusion over each image and generates the occluded images and error per joint and as average
import math
import torch
import argparse
import cv2
from models import hmr, SMPL
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

def get_occluded_imgs(batch, occ_size, occ_pixel, dataset_name, joint_idx, pred_camera):
    # Get the image batch find the ground truth joint location and occlude it
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gt_pose = batch['pose'].to(device)
    gt_betas = batch['betas'].to(device)
    images = batch['img'].to(device)
    gender = batch['gender'].to(device)
    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    # Preparing the regressor to map keypoints on to 14 joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    J_regressor = J_regressor.to(device)
    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Regressor broadcasting
    J_regressor_batch = J_regressor[None, :].expand(images.shape[0], -1, -1).to(device)
    # Get 14 ground truth joints
    if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
        gt_keypoints_3d = batch['pose_3d'].to(device)
        gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
    # For 3DPW get the 14 common joints from the rendered shape
    else:
        gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
        gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
        gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
        gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
        gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
        print(gt_keypoints_3d)
    # Project 3D keypoints to 2D keypoints
    camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    pred_keypoints_2d = perspective_projection(gt_keypoints_3d,
                                                   rotation=torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1),
                                                   translation=camera_translation,
                                                   focal_length=constants.FOCAL_LENGTH,
                                                   camera_center=camera_center)
    pred_keypoints_2d = pred_keypoints_2d[0].cpu().numpy()
    print(len(pred_keypoints_2d))
    # De-normalizing the image
    image = images
    image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
    img = image[0].permute(1,2,0).cpu().numpy()
    img = 255* img[:,:,::-1]
    print(img.shape)
    # for i in range(len(pred_keypoints_2d)):
    i=12
    cv2.circle(img, (int(pred_keypoints_2d[i][0]), int(pred_keypoints_2d[i][1])), 3, color = (255, 0, 0), thickness=-1)
    cv2.imwrite("test3.jpg", img)
    return 

def visualize_grid(image, heatmap, batch_idx):
    # Gets the image and heatmap and combine the two and show the result for each joint
    # image -> Torch.Size([1, 3, 224, 224])
    # heatmap -> (6, 6, 14)

    # De-normalizing the image
    image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)

    # Preparing the image for visualization
    image = image[0].permute(1,2,0).cpu().numpy()
    image = image[:,:,::-1]


    orig_heatmap = heatmap.copy()
    # Preparing the heatmap for visualization
    # This is how to change the size of the heatmap to cover the whole image
    # heatmap = resize(heatmap, image.shape)
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_CUBIC)
    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap) # normalize between [0,1]

    joint_names = ['Right Ankle',
                    'Right Knee',
                    'Right Hip',
                    'Left Hip',
                    'Left Knee',
                    'Left Ankle',
                    'Right Wrist',
                    'Right Elbow',
                    'Right Shoulder',
                    'Left Shoulder',
                    'Left Elbow',
                    'Left Wrist',
                    'Neck',
                    'Top of Head'
                    ]

    w, h = 7, 2
    f, axarr = plt.subplots(h, w)
    f.set_size_inches((w*3, h*3))
    # Show heatmap of per joint error on one separate image
    for jid in range(len(joint_names)):
        axarr[jid // w, jid % w].axis('off')
        axarr[jid // w, jid % w].set_title(
            f'{joint_names[jid]} \n'
            f'min: {orig_heatmap[:,:,jid].min()*1000:.2f} '
            f'max: {orig_heatmap[:,:,jid].max()*1000:.2f}'
        )
     
        axarr[jid // w, jid % w].imshow(image)
        axarr[jid // w, jid % w].imshow(heatmap[:,:,jid], alpha=0.3, cmap='jet', interpolation='none')

    f.set_tight_layout(tight=True)
    plt.savefig(os.path.join('examples/', f'result_00_{batch_idx:05d}_mpjpe_hm.png'))

def visualize_grid_mean(batch, heatmap, batch_idx, idx_dict, args):
    output_size = 480
    loader_size = batch['img'][0].shape[-1]
    img_orig = cv2.imread(batch['imgname'][0])
    center = batch['center'][0]
    scale = batch['scale'][0]
    img_orig = crop(img_orig, center, scale, (output_size, output_size))
    orig_heatmap = heatmap.copy()
    # Preparing the heatmap for visualization
    # This is how to change the size of the heatmap to cover the whole image
    heatmap = cv2.resize(heatmap, (img_orig.shape[0], img_orig.shape[1]), interpolation=cv2.INTER_CUBIC)
    # heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap) # normalize between [0,1]

    # Show the mean error of all joints on one image
    heatmap_mean = heatmap.mean(axis=2)
    heatmap_mean = 1000 * heatmap_mean

    # Put a box over the maximum
    heatmap_mean_org = orig_heatmap.mean(axis=2)
    heatmap_mean_org = 1000 * heatmap_mean_org
    h = idx_dict[np.argmax(heatmap_mean_org)][0]
    w = idx_dict[np.argmax(heatmap_mean_org)][1]
    img_size = int(img_orig.shape[0])
    occ_stride = int(round(args.stride * (output_size/loader_size)))
    occ_size = int(round(args.occ_size * (output_size/loader_size)))
    h_start = h * occ_stride
    w_start = w * occ_stride
    h_end = min(img_size, h_start + occ_size)
    w_end = min(img_size, w_start + occ_size)
    cv2.rectangle(img_orig, pt1=(w_start,h_start), pt2=(w_end,h_end), color=(255,0,0), thickness=2)

    plt.imshow(img_orig)
    plt.imshow(heatmap_mean, alpha=0.5, cmap='jet', interpolation='none')
    plt.colorbar()
    plt.savefig(os.path.join('examples/', f'result_00_{batch_idx:05d}_mpjpe_mean.png'))

def run_dataset(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the dataloader
    dataset_name = args.dataset
    occ_size = args.occ_size
    occ_pixel = args.pixel
    dataset = BaseDataset(None, dataset_name, is_train=False)
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # MPJPE error for the non-parametric and parametric shapes
    mpjpe_org = np.zeros(len(dataset))
    log_freq = args.log_freq
    # Load the model
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

    val_images_errors= []
    mpjpe = np.zeros(len(dataset))
    mpjpe_occluded = np.zeros(len(dataset))
    # mpjpe_per_joint = [[0] * 14] * (len(dataset))
    # for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
    batch_idx = 10
    batch = next(itertools.islice(data_loader, batch_idx, None))
    images = batch['img']
    curr_batch_size = images.shape[0]
    error, pred_camera = get_error(batch, model, dataset_name, args, smpl_neutral, smpl_male, smpl_female, J_regressor, joint_mapper_h36m, joint_mapper_gt)
    mpjpe[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] = error
    # Get occluded images
    joint_inx = 13
    images_occlude = get_occluded_imgs(batch, occ_size, occ_pixel, dataset_name, joint_inx, pred_camera)
        # batch['img'] = images_occlude
        # error = get_error(batch, model, dataset_name, args, smpl_neutral, smpl_male, smpl_female, J_regressor, joint_mapper_h36m, joint_mapper_gt)
        # mpjpe_occluded[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] = error

        # # Print intermediate results during evaluation
        # if batch_idx % log_freq == log_freq - 1:
        #     mpjpe =  np.array([i.mean() for i in mpjpe_per_joint[:batch_idx * batch_size]])
        #     print('MPJPE: ' + str(1000 * mpjpe[:batch_idx * batch_size].mean()))
        #     print()
    # Print final results during evaluation
    # print('*** Final Results ***')
    # print()
    # print('mpjpe: ' + str(1000 * mpjpe.mean()))
    # print()
    # print('mpjpe_occluded: ' + str(1000 * mpjpe_occluded.mean()))
    # print()

     


def get_error(batch, model, dataset_name, args, smpl_neutral, smpl_male, smpl_female,
                J_regressor, joint_mapper_h36m, joint_mapper_gt):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Get ground truth annotations from the batch
    gt_pose = batch['pose'].to(device)
    gt_betas = batch['betas'].to(device)
    gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
    images = batch['img'].to(device)
    gender = batch['gender'].to(device)
        
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(images)
        pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
    # Regressor broadcasting
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
    # Get 14 ground truth joints
    if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
        gt_keypoints_3d = batch['pose_3d'].to(device)
        gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
    # For 3DPW get the 14 common joints from the rendered shape
    else:
        gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
        gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
        gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
        gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
        gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
        gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    # Get 14 predicted joints from the mesh
    pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
    pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
    pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 
    # Absolute error (MPJPE)
    # error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
    error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    return error, pred_camera
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None) # Path to network checkpoint
    parser.add_argument('--dataset', type=str, default='3dpw')  # Path of the input image
    parser.add_argument('--occ_size', type=int, default='40')  # Size of occluding window
    parser.add_argument('--pixel', type=int, default='0')  # Occluding window - pixel values
    # parser.add_argument('--stride', type=int, default='20')  # Occlusion Stride
    parser.add_argument('--batch_size', default=1) # Batch size for testing
    parser.add_argument('--log_freq', default=50, type=int) # Frequency of printing intermediate results
    args = parser.parse_args()
    run_dataset(args)
    