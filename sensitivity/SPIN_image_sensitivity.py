# python3 sensitivity/SPIN_image_sensitivity.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --img_number=0
# This can run through whole dataset and moves the occlusion over each image and generates the occluded images and SPIN error per joint and as average
# Gets one image and moves the occluder
import sys
sys.path.insert(0, '/SPINH')
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

def get_occluded_imgs(img, occ_size, occ_pixel, occ_stride):
    # The input is the image in batch=1 and moves the occluder over the image and generate new occluded images
    # Input: torch.Size([1, 3, 224, 224])
    # output: torch.Size([36, 1, 3, 224, 224])
    img_size = int(img.shape[-1])
    # Define number of occlusions in both dimensions
    output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))

    occ_img_list = []

    idx_dict = {}
    c = 0
    for h in range(output_height):
        for w in range(output_width):
            # Occluder window:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

            # Getting the image copy, applying the occluding window:
            occ_image = img.clone()
            occ_image[:, :, h_start:h_end, w_start:w_end] = occ_pixel
            occ_img_list.append(occ_image)

            idx_dict[c] = (h,w)
            c += 1
    return torch.stack(occ_img_list, dim=0), idx_dict, output_height

def visualize_grid(image, heatmap, img_number):
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
    plt.savefig(os.path.join('sensitivity/', f'SPIN_result_00_{img_number:05d}_mpjpe_joint.png'))

def visualize_grid_mean(batch, heatmap, img_number, idx_dict, args):
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
    plt.colorbar(label="MPJPE (mm)")
    plt.savefig(os.path.join('sensitivity/', f'SPIN_result_00_{img_number:05d}_mpjpe_mean.png'))

def run_dataset(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the dataloader
    dataset_name = args.dataset
    dataset = BaseDataset(None, dataset_name, is_train=False)
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    
    # img_number = 21452
    img_number = args.img_number
    batch = next(itertools.islice(data_loader, img_number, None))
    print(batch['imgname'][0])
    images = batch['img']
    occluded_images, idx_dict, output_size = get_occluded_imgs(
        images,
        occ_size=args.occ_size,
        occ_pixel=args.pixel,
        occ_stride=args.stride
    )
    mpjpe_heatmap = np.zeros((output_size, output_size, 14))

    for occ_img_idx in tqdm(range(occluded_images.shape[0])):
        batch['img'] = occluded_images[occ_img_idx]
        per_mpjpe = get_error(batch, model, dataset_name, args, smpl_neutral, smpl_male, smpl_female, J_regressor, joint_mapper_h36m, joint_mapper_gt)
        mpjpe_heatmap[idx_dict[occ_img_idx]] = per_mpjpe[0]
    
    visualize_grid_mean(batch, mpjpe_heatmap, img_number, idx_dict, args)
    visualize_grid(images, mpjpe_heatmap, img_number)


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
    error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
    return error
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None) # Path to network checkpoint
    parser.add_argument('--dataset', type=str, default='3dpw')  # Path of the input image
    parser.add_argument('--occ_size', type=int, default='40')  # Size of occluding window
    parser.add_argument('--pixel', type=int, default='0')  # Occluding window - pixel values
    parser.add_argument('--stride', type=int, default='20')  # Occlusion Stride
    parser.add_argument('--batch_size', default=1) # Batch size for testing
    parser.add_argument('--log_freq', default=50, type=int) # Frequency of printing intermediate results
    parser.add_argument('--img_number', default=0, type=int) # the image number in 3DPW dataset that you want to investigate
    args = parser.parse_args()
    run_dataset(args)
    