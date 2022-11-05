# import glob
# import os

# image_list = glob.glob("/SPINH/data/H36M/images/*.jpg")
# print(len(image_list))
# for i in image_list:
#     os.rename(i, image_list[1000][:36] + "s" + image_list[1000][36:])

# import numpy as np
# import cv2
# from utils.imutils import crop

# data = np.load("data/dataset_extras/3dpw_test.npz")
# data = np.load("data/dataset_extras/h36m_valid_protocol1.npz")
# print(data.files)
# print(data['S'][0].shape)
# print(data['keypoints'][0])
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
import time
import itertools
from datasets import BaseDataset
from torch.utils.data import DataLoader
import torch
import cv2
import constants
from utils.imutils import transform
# from utils.geometry import perspective_projection
import numpy as np
from tqdm import tqdm
start = time.time()
# Load the dataloader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_name = '3dpw'
dataset = BaseDataset(None, dataset_name, is_train=False)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
#     print(batch['imgname'][-1])



batch_idx = 0
batch = next(itertools.islice(data_loader, batch_idx, None))
P = torch.matmul(batch['camera_intrinsics'], batch['camera_extrinsics'])
print(batch['imgname'])
joints = batch['joint_position']
joints = joints.reshape(-1, 24, 3)
batch_size = joints.shape[0]

# Preparing the regressor to map keypoints on to 14 joints
joint_mapper = [8, 5, 2, 1, 4, 7, 21, 19, 17,16, 18, 20, 12, 15]
# Get 14 ground truth joints
joints = joints[:, joint_mapper, :]
print(joints)

# Project 3D keypoints to 2D keypoints
temp = torch.ones((batch_size, joints.shape[1], 1)).double()
X = torch.cat((joints, temp), 2)
X = X.permute(0, 2, 1)
p = torch.matmul(P, X)
p = torch.div(p[:,:,:], p[:,2:3,:])
p = p[:, [0,1], :]
p = p.permute(0, 2, 1)


# Process 2d keypoints to match the process images in the dataset
center = batch['center']
scale = batch['scale']
res = [constants.IMG_RES, constants.IMG_RES]
new_p = np.ones((2,joints.shape[1],2))
for i in range(batch_size):
    for j in range(p.shape[1]):
        temp = transform(p[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
        new_p[i,j,:] = temp

# a = transform(p[i,j:j+1,:][0], center[i], scale[i], res, invert=0, rot=0)
# b = 2.*a/constants.IMG_RES - 1.
# b = b.astype('float32')


image = batch['img']
# De-normalizing the image
image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
# Preparing the image for visualization
img = image[0].permute(1,2,0).cpu().numpy().copy()
img = 255 * img[:,:,::-1]
for i in range(new_p.shape[1]):
    cv2.circle(img, (int(new_p[0][i][0]), int(new_p[0][i][1])), 3, color = (255, 0, 0), thickness=-1)
cv2.imwrite('test02.jpg', img)
end = time.time()
print("Time: ", end - start)



