# python3 sensitivity/OP_image_sensitivity.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --img_number=0
# This can run through whole dataset and moves the occlusion over each image and generates the occluded images and OpenPose error per joint and as average
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
from pytorchopenpose.src.body import Body
from utils.imutils import transform


def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

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
    plt.savefig(os.path.join('sensitivity/', f'OP_result_00_{batch_idx:05d}_mpjpe_joint.png'))

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
    plt.colorbar(label="MPJPE (mm)")
    plt.savefig(os.path.join('sensitivity/', f'OP_result_00_{batch_idx:05d}_mpjpe_mean.png'))

def run_dataset(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the dataloader
    dataset_name = args.dataset
    dataset = BaseDataset(None, dataset_name, is_train=False)
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # MPJPE error for the non-parametric and parametric shapes
    mpjpe_org = np.zeros(len(dataset))
    log_freq = args.log_freq
    # Load the model
    body_estimation_model = Body('pytorchopenpose/model/body_pose_model.pth')
    val_images_errors= []
    batch_idx = args.img_number
    # batch_idx = 0
    batch = next(itertools.islice(data_loader, batch_idx, None))
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
        per_mpjpe = get_error(batch, body_estimation_model)
        mpjpe_heatmap[idx_dict[occ_img_idx]] = per_mpjpe[0]
    
    visualize_grid_mean(batch, mpjpe_heatmap, batch_idx, idx_dict, args)
    visualize_grid(images, mpjpe_heatmap, batch_idx)
      


def get_error(batch, body_estimation_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Get the ground truth position of joints
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
    temp = torch.ones((batch_size, 14, 1)).double().to(device)
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
    gt_keypoints_2d = np.ones((batch_size,14,2))
    for i in range(batch_size):
        for j in range(p.shape[1]):
            temp = transform(p[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
            gt_keypoints_2d[i,j,:] = temp
    gt_keypoints_2d = torch.tensor(gt_keypoints_2d, device=device, dtype=torch.float)


    # Get the OpenPose prediction for joint positions
    occ_images = denormalize(batch['img'].to(device))
    curr_batch_size = occ_images.shape[0]
    candidate_sorted_list = []
    for i in range(curr_batch_size):
        candidate, subset = body_estimation_model(occ_images[i])
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
            error_s = torch.sqrt(((gt_keypoints_2d[i] - candidate_sorted_t) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            subset_error.append(error_s)
        subset_index = subset_error.index(min(subset_error))        
        subset_sorted = subset[subset_index][map_op_smpl].astype(int)
        candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
        candidate_sorted = candidate[subset_sorted]
        candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
        candidate_sorted_list.append(candidate_sorted_t)
    candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)

    # #test
    # img = batch['img']
    # img = denormalize(img.to(device))
    # # gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    # for i in range(candidate_sorted_t.shape[1]):
    #     cv2.circle(img[0], (int(candidate_sorted_t[0][i][0]), int(candidate_sorted_t[0][i][1])), 3, color = (255, 0, 0), thickness=1) #openpose
    # cv2.imwrite('examples/test.png', img[0])
    
    # Normalize between -1 and 1
    gt_keypoints_2d = torch.sub(gt_keypoints_2d, (constants.IMG_RES/2))
    gt_keypoints_2d = torch.div(gt_keypoints_2d, (constants.IMG_RES/2))
    # Relative position
    left_heap = gt_keypoints_2d[:,3:4,:].clone()
    left_heap = left_heap.expand(-1,14,-1)
    gt_keypoints_2d = gt_keypoints_2d - left_heap   
    # Normalize between -1 and 1
    candidate_sorted_t = torch.sub(candidate_sorted_t, (constants.IMG_RES/2))
    candidate_sorted_t = torch.div(candidate_sorted_t, (constants.IMG_RES/2))
    # Relative position
    left_heap = candidate_sorted_t[:,3:4,:].clone()
    left_heap = left_heap.expand(-1,14,-1)
    candidate_sorted_t = candidate_sorted_t - left_heap
    # Absolute error (MPJPE)
    error = torch.sqrt(((candidate_sorted_t - gt_keypoints_2d) ** 2).sum(dim=-1)).cpu().numpy()
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
    