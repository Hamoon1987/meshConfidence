# import glob
# import os

# image_list = glob.glob("/SPINH/data/H36M/images/*.jpg")
# print(len(image_list))
# for i in image_list:
#     os.rename(i, image_list[1000][:36] + "s" + image_list[1000][36:])

import numpy as np
# import cv2
# from utils.imutils import crop

data = np.load("data/dataset_extras/3dpw_test.npz")
# data = np.load("data/dataset_extras/h36m_valid_protocol2.npz")
# print(data.files)
# print(data['S'][0].shape)
# print(data['keypoints'][27900])
# for i in range(0, len(data['imgname']), 1000):
#     print(i)
#     img_path = "data/3DPW/" + data['imgname'][i]
#     print(img_path)
# img_path = "data/3DPW/" + data['imgname'][26893]
# print(img_path)
# img = cv2.imread(img_path)
# center = data['center'][0]
# scale = data['scale'][0]
# img = crop(img, center, scale, (224, 224))
# cv2.imwrite('test.jpg', img)

import itertools
from datasets import BaseDataset
from torch.utils.data import DataLoader
import torch
# import cv2
# import constants
# from utils.geometry import perspective_projection
import numpy as np
from tqdm import tqdm
# Load the dataloader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_name = '3dpw'
dataset = BaseDataset(None, dataset_name, is_train=False)
batch_size = 1000
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
    print(batch['imgname'][-1])



# batch_idx = 8000-810
# batch = next(itertools.islice(data_loader, batch_idx, None))

# img = batch['img']
# keypoints = batch['pose']


# # preparing the image
# from utils.imutils import crop
# img_orig = cv2.imread(batch['imgname'][0])
# output_size = 224
# print(batch['imgname'][0])
# center = batch['center'][0]
# scale = batch['scale'][0]
# img_crop = crop(img_orig, center, scale, (output_size, output_size))
# cv2.imwrite('./examples/image_00000.jpg', img_crop)


# camera_translation = batch['translation'].to(device)
# # #preparing 3D keypoints
# keypoints = torch.reshape(keypoints, (-1, 24, 3))
# joint_mapper_h36m = constants.H36M_TO_J17
# keypoints = keypoints[:, joint_mapper_h36m, :]
# keypoints = keypoints [:, :14, :].to(device)
# # keypoints = 255 * keypoints + 112
# print(keypoints)


# # Projecting 3D keypoints to 2D keypoints

# camera_center = torch.tensor([constants.IMG_RES // 2, constants.IMG_RES // 2])
# pred_keypoints_2d = perspective_projection(keypoints,
#                                                 rotation=torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1),
#                                                 translation=camera_translation,
#                                                 focal_length=constants.FOCAL_LENGTH,
#                                                 camera_center=camera_center)
# pred_keypoints_2d = pred_keypoints_2d[0].cpu().numpy()



# # Preparing the image for visualization
# img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1, 3, 1, 1)
# img = img + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1, 3, 1, 1)
# img = img[0].permute(1,2,0).cpu().numpy()
# img = 255 * img[:,:,::-1]
# imgc = img.copy()

# for i in range(14):
#     # cv2.circle(imgc, (int(pred_keypoints_2d[i][0]), int(pred_keypoints_2d[i][1])), 5, color = (255, 0, 0), thickness=-1)
#     cv2.circle(imgc, (int(keypoints[0][i][0]), int(keypoints[0][i][1])), 5, color = (255, 0, 0), thickness=-1)


# cv2.imwrite('test.jpg', imgc)


# # img = cv2.imread(batch['imgname'][0])

# # print(batch['img'][0])
# # print(batch['pose'][0])
