# python3 sp_op/h36m_test.py --checkpoint=/SPINH/data/model_checkpoint.pt --dataset=h36m-p2 --log_freq=20


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
import argparse
from tqdm import tqdm
from utils.imutils import transform
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p2', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=20 , type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=1, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes for data loading')


def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

def normalize(data):
        data_n = torch.sub(data, (constants.IMG_RES/2))
        data_n = torch.div(data_n, (constants.IMG_RES/2))
        return data_n

def run_evaluation(model, joint_index, dataset_name, dataset,
                   batch_size=16, img_res=224, 
                   num_workers=1, shuffle=False, log_freq=20):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
    # Transfer model to the GPU
    model.to(device)
    sp_gt = np.zeros(len(dataset))
    sp_op = np.zeros(len(dataset))
    op_conf = np.zeros(len(dataset))

    # for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
    step = 0
    batch = next(itertools.islice(data_loader, step, None))
    
    images = batch['img'].to(device)
    curr_batch_size = images.shape[0]

    # 2D GT from labels
    S_2D = batch['S_2D']
    keywords_map=[0,1,2,3,4,5,6,7,8,9,10,11,12,17,16] # Spine is 16
    S_2D = S_2D[:, keywords_map,:2] * 1000 # 1000 is for denormalizing (Initial size 1000)
    center = batch['center']
    scale = batch['scale']
    res = [constants.IMG_RES, constants.IMG_RES]
    gt_keypoints_2d = np.ones((curr_batch_size,S_2D.shape[1],2))
    for i in range(curr_batch_size):
        for j in range(S_2D.shape[1]):
            temp = transform(S_2D[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
            gt_keypoints_2d[i,j,:] = temp
    gt_keypoints_2d = torch.tensor(gt_keypoints_2d, dtype=torch.float).to(device)
    gt_spine_2d = gt_keypoints_2d[:, [-1], :].clone()

    # 2D predicted keypoint
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(images)
        pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        smpl_pred_joints = pred_output.joints
    # pred_pelvis = smpl_pred_joints[:, [39],:].clone()
    # smpl_pred_joints = smpl_pred_joints - pred_pelvis
    # 2D projection of SPIN pred_keypoints and gt_keypoints
    focal_length = constants.FOCAL_LENGTH
    camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    smpl_pred_keypoints_2d = perspective_projection(smpl_pred_joints,
                                        rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                        translation=camera_translation,
                                        focal_length=focal_length,
                                        camera_center=camera_center)
    pred_spine_2d = smpl_pred_keypoints_2d[:, [41],:].clone()
    smpl_pred_keypoints_2d = smpl_pred_keypoints_2d - pred_spine_2d + gt_spine_2d
    smpl_joint_map_op = [11, 10, 27, 28, 13, 14, 4, 3, 2, 5, 6, 7, 40, 0, 41] # 41 is spine
    smpl_joint_map_gt = [11, 10, 27, 28, 13, 14, 4, 3, 2, 5, 6, 7, 37, 42] # 39 is pelvis
    smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d[:, smpl_joint_map_op, :]
    smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d[:, smpl_joint_map_gt, :]
    # smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d_gt - pred_pelvis_2d + gt_pelvis_2d

    
    # OpenPose keypoints
    body_estimation = Body('pytorchopenpose/model/body_pose_model.pth')
    image_ = denormalize(images)
    candidate_sorted_list = []
    op_confidence_list = []
    for i in range(curr_batch_size):
        candidate, subset = body_estimation(image_[i])
        map_op_smpl = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 0]
        if subset.shape[0] == 0:
            a = np.zeros((14,2))
            b = np.zeros((14,1))
            candidate_sorted_list.append(torch.tensor(a, dtype=torch.float))
            op_confidence_list.append(torch.tensor(b, dtype=torch.float))
            continue
        subset_sorted = subset[0][map_op_smpl].astype(int)
        candidate = np.vstack([candidate, [constants.IMG_RES/2, constants.IMG_RES/2, 0, -1]])
        candidate_sorted = candidate[subset_sorted]
        op_confidence = torch.tensor(candidate_sorted[:,2:3], dtype=torch.float)
        candidate_sorted_t = torch.tensor(candidate_sorted[:,:2], dtype=torch.float)
        candidate_sorted_list.append(candidate_sorted_t)
        op_confidence_list.append(op_confidence)
    candidate_sorted_t = torch.stack(candidate_sorted_list, dim=0).to(device)
    op_confidence_t = torch.stack(op_confidence_list, dim=0).to(device)

    op_spine = (((candidate_sorted_t[:, [2], :] + candidate_sorted_t[:, [3], :]) / 2) + candidate_sorted_t[:, [12], :]) / 2
    candidate_sorted_t = candidate_sorted_t - op_spine + gt_spine_2d

    candidate_sorted_t_n = normalize(candidate_sorted_t)
    smpl_pred_keypoints_2d_op_n =normalize(smpl_pred_keypoints_2d_op)
    smpl_pred_keypoints_2d_gt_n = normalize(smpl_pred_keypoints_2d_gt)
    gt_keypoints_2d_n = normalize(gt_keypoints_2d)

    # # Absolute error SPIN (MPJPE)
    error = torch.sqrt(((smpl_pred_keypoints_2d_gt_n[:, joint_index, :] - gt_keypoints_2d_n[:, joint_index, :]) ** 2).sum(dim=-1)).cpu().numpy()
    print("SP-GT", error)
    sp_gt[step * batch_size:step * batch_size + curr_batch_size] = error
    # SPIN - OpenPose (sp_op)
    error_ = torch.sqrt(((smpl_pred_keypoints_2d_op_n[:, joint_index, :] - candidate_sorted_t_n[:, joint_index, :]) ** 2).sum(dim=-1)).cpu().numpy()
    print("SP-OP", error_)
    sp_op[step * batch_size:step * batch_size + curr_batch_size] = error_
    # OpenPose Confidence
    op_confidence_joint = op_confidence_t[:, joint_index, 0].cpu().numpy()
    op_conf[step * batch_size:step * batch_size + curr_batch_size] = op_confidence_joint

    # # Print intermediate results during evaluation
    # if step % log_freq == log_freq - 1:
    #     print('sp_gt: ' + str(1000 * sp_gt[:step * batch_size].mean()))
    #     print('sp_op: ' + str(1000 * sp_op[:step * batch_size].mean()))
    #     print()


    # Visualize
    candidate_sorted_t = candidate_sorted_t[0]
    gt_keypoints_2d = gt_keypoints_2d[0]
    smpl_pred_keypoints_2d = smpl_pred_keypoints_2d[0]
    smpl_pred_keypoints_2d_gt = smpl_pred_keypoints_2d_gt[0]
    smpl_pred_keypoints_2d_op = smpl_pred_keypoints_2d_op[0]
    image_test = image_[0]
    for i in range(14):
        # i = joint_index
        cv2.circle(image_test, (int(gt_keypoints_2d[i][0]), int(gt_keypoints_2d[i][1])), 3, color = (0, 255, 0), thickness=-1)
        cv2.circle(image_test, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 2, color = (0, 0, 255), thickness=-1)
        cv2.circle(image_test, (int(smpl_pred_keypoints_2d_gt[i][0]), int(smpl_pred_keypoints_2d_gt[i][1])), 2, color = (255, 0, 0), thickness=-1)
        cv2.circle(image_test, (int(smpl_pred_keypoints_2d_op[i][0]), int(smpl_pred_keypoints_2d_op[i][1])), 2, color = (255, 255, 255), thickness=-1)
        #    
    # for i in range(14):
    #     # i=43
    #     cv2.circle(image_test, (int(candidate_sorted_t[i][0]), int(candidate_sorted_t[i][1])), 2, color = (0, 0, 255), thickness=-1)
    #     cv2.circle(image_test, (int(smpl_pred_keypoints_2d_gt[i][0]), int(smpl_pred_keypoints_2d_gt[i][1])), 2, color = (255, 0, 0), thickness=-1)
        
    cv2.imwrite(f'sp_op/h36_test.png', image_test)


if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    print(len(dataset))
    # Run evaluation
    for i in range(12,13):
        print("joint_index = ", i)
        joint_index = i    
        run_evaluation(model, joint_index, args.dataset, dataset,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    log_freq=args.log_freq)